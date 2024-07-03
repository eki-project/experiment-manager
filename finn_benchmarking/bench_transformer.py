import os
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import json
import time
import math
from shutil import copy as shcopy
import copy
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.custom_op.general.multithreshold import multithreshold

from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import (
    calculate_matvec_accumulator_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)
from qonnx.util.basic import roundup_to_integer_multiple
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

import finn.core.onnx_exec as oxe

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext
from finn.transformation.fpgadataflow.make_zynq_proj import collect_ip_dirs
from finn.util.basic import make_build_dir, pynq_native_port_width, pynq_part_map

# YAML for loading experiment configurations
import yaml
import pandas as pd
# FINN dataflow builder
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import AutoFIFOSizingMethod

# Custom build steps required to streamline and convert the attention operator
from transformer_build_steps import (
    step_tidy_up_pre_attention,
    step_tidy_up_post_attention,
    step_streamline_attention,
    step_streamline_residual,
    step_streamline_norms,
    step_streamline_positional,
    step_convert_attention_to_hw,
    step_convert_elementwise_binary_to_hw,
    step_replicate_streams,
    set_target_parallelization,
    set_fifo_depths,
    step_apply_folding_config,
    node_by_node_rtlsim,
    node_by_node_cppsim
)

# power report scripting based on Lucas Reuter:
template_open = """
open_project  $PROJ_PATH$
open_run $RUN$
"""

template_single_test = """
set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type lut [get_cells -r finn_design_i/.*]
set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type register [get_cells -r finn_design_i/.*]
set_switching_activity -deassert_resets
report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
reset_switching_activity -hier -type lut [get_cells -r finn_design_i/.*]
reset_switching_activity -hier -type register [get_cells -r finn_design_i/.*]
"""

#template_single_test_type = """
#set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type $SWITCH_TARGET$ [get_cells -r finn_design_i/.*]
#set_switching_activity -deassert_resets
#report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
#reset_switching_activity -hier -type $SWITCH_TARGET$ [get_cells -r finn_design_i/.*]
#"""

template_sim_power = """
set_property SOURCE_SET sources_1 [get_filesets sim_1]
import_files -fileset sim_1 -norecurse $TB_FILE_PATH$
set_property top switching_simulation_tb [get_filesets sim_1]
update_compile_order -fileset sim_1

launch_simulation -mode post-implementation -type functional
restart
open_saif $SAIF_FILE_PATH$
log_saif [get_objects -r /switching_simulation_tb/dut/*]
run $SIM_DURATION_NS$ ns
close_saif

read_saif $SAIF_FILE_PATH$
report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
"""

template_switching_simulation_tb = """
`timescale 1 ns/10 ps

module switching_simulation_tb;
reg clk;
reg rst;

//dut inputs
reg tready;
reg [$INSTREAM_WIDTH$-1:0] tdata;
reg tvalid;

//dut outputs
wire [$OUTSTREAM_WIDTH$-1:0] accel_tdata;
wire accel_tready;
wire accel_tvalid; 

finn_design_wrapper dut(
        .ap_clk(clk), 
        .ap_rst_n(rst),
        .m_axis_0_tdata(accel_tdata),
        .m_axis_0_tready(tready),
        .m_axis_0_tvalid(accel_tvalid),
        .s_axis_0_tdata(tdata),
        .s_axis_0_tready(accel_tready),
        .s_axis_0_tvalid(tvalid)
        );   

always 
    begin
        clk = 0;
        #2.5;
        clk = 1;
        #2.5;
    end

integer i;
initial
    begin
        tready = 0;
        tdata = 0;
        tvalid = 0;
        rst = 0;
        #50;
        rst = 1;
        tvalid = 1;
        tready = 1;
        while(1)
            begin
                for (i = 0; i < $INSTREAM_WIDTH$/$DTYPE_WIDTH$; i = i+1) begin
                    tdata[i*$DTYPE_WIDTH$ +: $DTYPE_WIDTH$] = $RANDOM_FUNCTION$;
                end
                #5;
            end
    end
endmodule
"""

zynq_harness_template = """
set FREQ_MHZ %s
set NUM_AXILITE %d
if {$NUM_AXILITE > 9} {
    error "Maximum 10 AXI-Lite interfaces supported"
}
set NUM_AXIMM %d
set BOARD %s
set FPGA_PART %s
create_project finn_zynq_link ./ -part $FPGA_PART

# set board part repo paths to find boards installed by FINN
set paths_prop [get_property BOARD_PART_REPO_PATHS [current_project]]
set paths_param [get_param board.repoPaths]
lappend paths_prop $::env(FINN_ROOT)/deps/board_files
lappend paths_param $::env(FINN_ROOT)/deps/board_files
set_property BOARD_PART_REPO_PATHS $paths_prop [current_project]
set_param board.repoPaths $paths_param

if {$BOARD == "RFSoC2x2"} {
    set_property board_part xilinx.com:rfsoc2x2:part0:1.1 [current_project]
    set ZYNQ_TYPE "zynq_us+"
} else {
    puts "Unrecognized board"
}

create_bd_design "top"
if {$ZYNQ_TYPE == "zynq_us+"} {
    set zynq_ps_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:zynq_ultra_ps_e:*"]]
    create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ps
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ps]
    set_property CONFIG.PSU__DISPLAYPORT__PERIPHERAL__ENABLE {0} [get_bd_cells zynq_ps]
    #activate one slave port, deactivate the second master port
    set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 {0}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__USE__M_AXI_GP1 {0}] [get_bd_cells zynq_ps]
    #set frequency of PS clock (this can't always be exactly met)
    set_property -dict [list CONFIG.PSU__OVERRIDE__BASIC_CLOCK {0}] [get_bd_cells zynq_ps]
    set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [expr int($FREQ_MHZ)]] [get_bd_cells zynq_ps]
} else {
    puts "Unrecognized Zynq type"
}

#instantiate axi interconnect, axi smartconnect
set interconnect_vlnv [get_property VLNV [get_ipdefs -all "xilinx.com:ip:axi_interconnect:*" -filter design_tool_contexts=~*IPI*]]
#set smartconnect_vlnv [get_property VLNV [get_ipdefs "xilinx.com:ip:smartconnect:*"]]
create_bd_cell -type ip -vlnv $interconnect_vlnv axi_interconnect_0
#create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
#set number of axilite interfaces, and number of axi master interfaces
#set_property -dict [list CONFIG.NUM_SI $NUM_AXIMM] [get_bd_cells smartconnect_0]
set_property -dict [list CONFIG.NUM_MI $NUM_AXILITE] [get_bd_cells axi_interconnect_0]

#create reset controller and connect interconnects to PS
if {$ZYNQ_TYPE == "zynq_us+"} {
    set axi_peripheral_base 0xA0000000
    #connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]
    connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
    #connect interconnect clocks and resets
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/ACLK]
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins axi_interconnect_0/S00_ACLK]
    #apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} Freq {} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ps/saxihp0_fpd_aclk]
}
#connect_bd_net [get_bd_pins axi_interconnect_0/ARESETN] [get_bd_pins smartconnect_0/aresetn]

#procedure used by below IP instantiations to map BD address segments based on the axi interface aperture
proc assign_axi_addr_proc {axi_intf_path} {
    #global variable holds current base address
    global axi_peripheral_base
    #infer range
    set range [expr 2**[get_property CONFIG.ADDR_WIDTH [get_bd_intf_pins $axi_intf_path]]]
    set range [expr $range < 4096 ? 4096 : $range]
    #align base address to range
    set offset [expr ($axi_peripheral_base + ($range-1)) & ~($range-1)]
    #perform assignment
    assign_bd_address [get_bd_addr_segs $axi_intf_path/Reg*] -offset $offset -range $range
    #advance base address
    set axi_peripheral_base [expr $offset + $range]
}

#custom IP instantiations/connections start here
%s

#finalize clock and reset connections for interconnects
if {$ZYNQ_TYPE == "zynq_us+"} {
    apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ps/pl_clk0} }  [get_bd_pins axi_interconnect_0/M*_ACLK]
}

save_bd_design
assign_bd_address
validate_bd_design

set_property SYNTH_CHECKPOINT_MODE "Hierarchical" [ get_files top.bd ]
make_wrapper -files [get_files top.bd] -import -fileset sources_1 -top

#set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
#set_property strategy Performance_ExtraTimingOpt [get_runs impl_1]
#set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
#set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
#set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
#set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]

# out-of-context synth can't be used for bitstream generation
# set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} -objects [get_runs synth_1]
launch_runs -to_step write_bitstream impl_1
wait_on_run [get_runs impl_1]

# generate synthesis report
open_run impl_1
report_utilization -hierarchical -hierarchical_depth 4 -file synth_report.xml -format xml
close_project
"""

class MakeZYNQHarnessProject(Transformation):
    """ Based on MakeZYNQProject transformation, but integrates IP into test harness instead of DMA shell."""

    def __init__(self, platform, results_dir, run_id, dut_duplication=1):
        super().__init__()
        self.platform = platform
        self.results_dir = results_dir
        self.run_id = run_id
        self.dut_duplication = dut_duplication

    def apply(self, model):
        # create a config file and empty list of xo files
        config = []
        idma_idx = 0
        odma_idx = 0
        aximm_idx = 0
        axilite_idx = 0
        global_clk_ns = 0

        # assume single stitched-ip (previously dataflowpartition) as DUT
        node = model.graph.node[0]
        node_inst = getCustomOp(node)
        instream_width = node_inst.get_instream_width_padded()
        outstream_width = node_inst.get_outstream_width_padded()
        #TODO: make compatible with multi-layer DUTs

        #assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
        #sdp_node = getCustomOp(node)
        #dataflow_model_filename = sdp_node.get_nodeattr("model")
        #kernel_model = ModelWrapper(dataflow_model_filename)
        kernel_model = model

        ipstitch_path = kernel_model.get_metadata_prop("vivado_stitch_proj")
        if ipstitch_path is None or (not os.path.isdir(ipstitch_path)):
            raise Exception(
                "No stitched IPI design found for %s, apply CreateStitchedIP first." % node.name
            )

        vivado_stitch_vlnv = kernel_model.get_metadata_prop("vivado_stitch_vlnv")
        if vivado_stitch_vlnv is None:
            raise Exception("No vlnv found for %s, apply CreateStitchedIP first." % node.name)

        ip_dirs = ["list"]
        ip_dirs += collect_ip_dirs(kernel_model, ipstitch_path)
        ip_dirs.append("$::env(FINN_ROOT)/../power_measurement/harness_sink/ip")
        ip_dirs_str = "[%s]" % (" ".join(ip_dirs))
        config.append(
            "set_property ip_repo_paths "
            "[concat [get_property ip_repo_paths [current_project]] %s] "
            "[current_project]" % ip_dirs_str
        )
        config.append("update_ip_catalog -rebuild -scan_changes")
        config.append("import_files -fileset sources_1 -norecurse $::env(FINN_ROOT)/../power_measurement/vector_xor.v")

        # get metadata property clk_ns to calculate clock frequency
        clk_ns = float(kernel_model.get_metadata_prop("clk_ns"))
        if clk_ns > global_clk_ns:
            global_clk_ns = clk_ns

        ifnames = eval(kernel_model.get_metadata_prop("vivado_stitch_ifnames"))

        # instantiate DUT, TODO: switch to wrapper verilog file for (multiple-) DUT instantiation
        for id in range(self.dut_duplication):
            dut_instance_name = "finn_design_%d"%id
            config.append("create_bd_cell -type ip -vlnv %s %s" % (vivado_stitch_vlnv, dut_instance_name))
            #sdp_node.set_nodeattr("instance_name", instance_names[node.name])
            config.append("connect_bd_net [get_bd_pins %s/ap_clk] [get_bd_pins axi_interconnect_0/aclk]" % dut_instance_name)
            config.append("connect_bd_net [get_bd_pins %s/ap_rst_n] [get_bd_pins axi_interconnect_0/aresetn]" % dut_instance_name)

        # instantiate input harness
        if instream_width > 8192:
            print("ERROR: DUT input stream width > 8192")
            raise Exception("ERROR: DUT input stream width > 8192")
        elif instream_width > 4096:
            num_sources = 8
            source_width = roundup_to_integer_multiple(instream_width/8, 8)
        elif instream_width > 2048:
            num_sources = 4
            source_width = roundup_to_integer_multiple(instream_width/4, 8)
        elif instream_width > 1024:
            num_sources = 2
            source_width = roundup_to_integer_multiple(instream_width/2, 8)
        else:
            num_sources = 1
            source_width = instream_width

        if self.dut_duplication > 1:
            if num_sources > 1:
                print("ERROR: DUT duplication with >1024 stream width not supported!")
                raise Exception("ERROR: DUT duplication with >1024 stream width not supported!")
            
            num_sources = self.dut_duplication # one source per DUT instance
            seed = 0xABCD
            for id in range(num_sources):
                config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axi_traffic_gen:3.0 axi_traffic_gen_%d"%id)
                config.append("set_property -dict [list \
                    CONFIG.C_ATG_MODE {AXI4-Stream} \
                    CONFIG.C_ATG_STREAMING_MAX_LEN_BITS {1} \
                    CONFIG.C_AXIS_SPARSE_EN {false} \
                    CONFIG.C_AXIS_TDATA_WIDTH {%d} \
                    CONFIG.C_AXIS_TDEST_WIDTH {0} \
                    CONFIG.C_AXIS_TID_WIDTH {0} \
                    CONFIG.C_AXIS_TUSER_WIDTH {0} \
                    CONFIG.STRM_DATA_SEED {%s} \
                    ] [get_bd_cells axi_traffic_gen_%d]"%(source_width,"0x{:04X}".format(seed),id))
                config.append("connect_bd_net [get_bd_pins axi_traffic_gen_%d/s_axi_aclk] [get_bd_pins axi_interconnect_0/aclk]"%id)
                config.append("connect_bd_net [get_bd_pins axi_traffic_gen_%d/s_axi_aresetn] [get_bd_pins axi_interconnect_0/aresetn]"%id)
                seed = seed + 99

                config.append("connect_bd_intf_net [get_bd_intf_pins axi_traffic_gen_%d/M_AXIS_MASTER] [get_bd_intf_pins finn_design_%d/s_axis_0]"%(id,id))

        else:
            seed = 0xABCD
            for id in range(num_sources):
                config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axi_traffic_gen:3.0 axi_traffic_gen_%d"%id)
                config.append("set_property -dict [list \
                    CONFIG.C_ATG_MODE {AXI4-Stream} \
                    CONFIG.C_ATG_STREAMING_MAX_LEN_BITS {1} \
                    CONFIG.C_AXIS_SPARSE_EN {false} \
                    CONFIG.C_AXIS_TDATA_WIDTH {%d} \
                    CONFIG.C_AXIS_TDEST_WIDTH {0} \
                    CONFIG.C_AXIS_TID_WIDTH {0} \
                    CONFIG.C_AXIS_TUSER_WIDTH {0} \
                    CONFIG.STRM_DATA_SEED {%s} \
                    ] [get_bd_cells axi_traffic_gen_%d]"%(source_width,"0x{:04X}".format(seed),id))
                config.append("connect_bd_net [get_bd_pins axi_traffic_gen_%d/s_axi_aclk] [get_bd_pins axi_interconnect_0/aclk]"%id)
                config.append("connect_bd_net [get_bd_pins axi_traffic_gen_%d/s_axi_aresetn] [get_bd_pins axi_interconnect_0/aresetn]"%id)
                config.append("connect_bd_net [get_bd_pins finn_design_0/s_axis_0_tready] [get_bd_pins axi_traffic_gen_%d/m_axis_1_tready]"%id)
                seed = seed + 99
    
            if num_sources > 1:
                config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_tdata")
                config.append("set_property CONFIG.NUM_PORTS {%d} [get_bd_cells xlconcat_tdata]"%num_sources)

                for id in range(num_sources):
                    config.append("connect_bd_net [get_bd_pins xlconcat_tdata/In%d] [get_bd_pins axi_traffic_gen_%d/m_axis_1_tdata]"%(id, id))

                config.append("connect_bd_net [get_bd_pins finn_design_0/s_axis_0_tdata] [get_bd_pins xlconcat_tdata/dout]")
            else:
                config.append("connect_bd_net [get_bd_pins finn_design_0/s_axis_0_tdata] [get_bd_pins axi_traffic_gen_0/m_axis_1_tdata]")
                
            # only connect valid from source 0 to DUT
            config.append("connect_bd_net [get_bd_pins finn_design_0/s_axis_0_tvalid] [get_bd_pins axi_traffic_gen_0/m_axis_1_tvalid]")
            

        # instantiate output harness
        for id in range(self.dut_duplication):
            config.append("create_bd_cell -type ip -vlnv xilinx.com:user:harness_sink:1.0 sink_%d"%id)
            config.append("set_property -dict [list CONFIG.STREAM_WIDTH {%d}] [get_bd_cells sink_%d]"%(outstream_width,id))
            config.append("connect_bd_intf_net [get_bd_intf_pins sink_%d/s_axis_0] [get_bd_intf_pins finn_design_%d/m_axis_0]"%(id,id))

        # GPIO control (TODO: connect interrupt)
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0")
        config.append("set_property -dict [list \
            CONFIG.C_ALL_INPUTS {0} \
            CONFIG.C_GPIO_WIDTH {5} \
            CONFIG.C_INTERRUPT_PRESENT {1} \
            ] [get_bd_cells axi_gpio_0]")
        config.append(
            "connect_bd_intf_net [get_bd_intf_pins axi_gpio_0/S_AXI] "
            "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]"
            % (axilite_idx)
        )
        config.append("assign_axi_addr_proc axi_gpio_0/S_AXI")
        axilite_idx += 1
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_0")
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_1")
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 xlslice_2")
        config.append("set_property -dict [list \
            CONFIG.DIN_FROM {0} \
            CONFIG.DIN_TO {0} \
            CONFIG.DIN_WIDTH {5} \
            ] [get_bd_cells xlslice_0]")
        config.append("set_property -dict [list \
            CONFIG.DIN_FROM {1} \
            CONFIG.DIN_TO {1} \
            CONFIG.DIN_WIDTH {5} \
            ] [get_bd_cells xlslice_1]")
        config.append("set_property -dict [list \
            CONFIG.DIN_FROM {2} \
            CONFIG.DIN_TO {2} \
            CONFIG.DIN_WIDTH {5} \
            ] [get_bd_cells xlslice_2]")
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0")
        config.append("set_property -dict [list CONFIG.IN1_WIDTH.VALUE_SRC USER CONFIG.IN2_WIDTH.VALUE_SRC USER CONFIG.IN0_WIDTH.VALUE_SRC USER] [get_bd_cells xlconcat_0]")
        config.append("set_property -dict [list \
            CONFIG.IN0_WIDTH {3} \
            CONFIG.NUM_PORTS {3} \
            ] [get_bd_cells xlconcat_0]")
        config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 xlconstant_0")
        config.append("set_property -dict [list \
            CONFIG.CONST_VAL {0} \
            CONFIG.CONST_WIDTH {3} \
            ] [get_bd_cells xlconstant_0]")
        config.append("""
            connect_bd_net [get_bd_pins xlslice_0/Din] [get_bd_pins axi_gpio_0/gpio_io_o]
            connect_bd_net [get_bd_pins xlslice_1/Din] [get_bd_pins axi_gpio_0/gpio_io_o]
            connect_bd_net [get_bd_pins xlslice_2/Din] [get_bd_pins axi_gpio_0/gpio_io_o]
            connect_bd_net [get_bd_pins xlconstant_0/dout] [get_bd_pins xlconcat_0/In0]
            connect_bd_net [get_bd_pins axi_gpio_0/gpio_io_i] [get_bd_pins xlconcat_0/dout]
        """)
        if self.dut_duplication > 1:
            config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_valid")
            config.append("set_property CONFIG.NUM_PORTS {%d} [get_bd_cells xlconcat_valid]"%self.dut_duplication)
            config.append("create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_checksum")
            config.append("set_property CONFIG.NUM_PORTS {%d} [get_bd_cells xlconcat_checksum]"%self.dut_duplication)

            config.append("create_bd_cell -type module -reference vector_xor vector_xor_valid")
            config.append("set_property CONFIG.WIDTH {%d} [get_bd_cells vector_xor_valid]"%self.dut_duplication)
            config.append("create_bd_cell -type module -reference vector_xor vector_xor_checksum")
            config.append("set_property CONFIG.WIDTH {%d} [get_bd_cells vector_xor_checksum]"%self.dut_duplication)

            config.append("connect_bd_net [get_bd_pins vector_xor_valid/in_data] [get_bd_pins xlconcat_valid/dout]")
            config.append("connect_bd_net [get_bd_pins vector_xor_checksum/in_data] [get_bd_pins xlconcat_checksum/dout]")
            config.append("connect_bd_net [get_bd_pins vector_xor_valid/out_data] [get_bd_pins xlconcat_0/In1]")
            config.append("connect_bd_net [get_bd_pins vector_xor_checksum/out_data] [get_bd_pins xlconcat_0/In2]")
            for id in range(self.dut_duplication):
                config.append("connect_bd_net [get_bd_pins sink_%d/valid] [get_bd_pins xlconcat_valid/In%d]"%(id,id))
                config.append("connect_bd_net [get_bd_pins sink_%d/checksum] [get_bd_pins xlconcat_checksum/In%d]"%(id,id))
        else:
            config.append("connect_bd_net [get_bd_pins sink_0/valid] [get_bd_pins xlconcat_0/In1]")
            config.append("connect_bd_net [get_bd_pins sink_0/checksum] [get_bd_pins xlconcat_0/In2]")
        for id in range(self.dut_duplication):
            config.append("connect_bd_net [get_bd_pins xlslice_2/Dout] [get_bd_pins sink_%d/enable]"%id)
        for id in range(num_sources):
            config.append("connect_bd_net [get_bd_pins xlslice_0/Dout] [get_bd_pins axi_traffic_gen_%d/core_ext_start]"%id)
            config.append("connect_bd_net [get_bd_pins xlslice_1/Dout] [get_bd_pins axi_traffic_gen_%d/core_ext_stop]"%id)

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_zynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        fclk_mhz = int(1 / (global_clk_ns * 0.001))

        # create a TCL recipe for the project
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        config = "\n".join(config) + "\n"
        with open(ipcfg, "w") as f:
            f.write(
                zynq_harness_template
                % (
                    fclk_mhz,
                    axilite_idx,
                    aximm_idx,
                    self.platform,
                    pynq_part_map[self.platform],
                    config,
                )
            )

        # create a TCL recipe for the project
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        working_dir = os.environ["PWD"]
        with open(synth_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_pynq_proj_dir))
            f.write("vivado -mode batch -source %s\n" % ipcfg)
            f.write("cd {}\n".format(working_dir))

        # call the synthesis script
        bash_command = ["bash", synth_project_sh]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()

        # collect results
        results_dir_bitstreams = os.path.join(self.results_dir, "bitstreams")
        os.makedirs(results_dir_bitstreams, exist_ok=True)

        bitfile_name = vivado_pynq_proj_dir + "/finn_zynq_link.runs/impl_1/top_wrapper.bit"
        if not os.path.isfile(bitfile_name):
            raise Exception("Synthesis failed, no bitfile found. Check logs under %s" % vivado_pynq_proj_dir)
        deploy_bitfile_name = results_dir_bitstreams + "/run_%d.bit"%self.run_id
        shcopy(bitfile_name, deploy_bitfile_name)

        hwh_name = vivado_pynq_proj_dir + "/finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh"
        if not os.path.isfile(hwh_name):
            raise Exception("Synthesis failed, no hwh file found. Check logs under %s" % vivado_pynq_proj_dir)
        deploy_hwh_name = results_dir_bitstreams + "/run_%d.hwh"%self.run_id
        shcopy(hwh_name, deploy_hwh_name)

        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        deploy_synth_report_filename = results_dir_bitstreams + "/run_%d.xml"%self.run_id
        shcopy(synth_report_filename, deploy_synth_report_filename)

        #model.set_metadata_prop("bitfile", deploy_bitfile_name)
        #model.set_metadata_prop("hw_handoff", deploy_hwh_name)
        #model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        return (model, False)

def _find_rows_and_headers(table):
    rows = table.findall("tablerow")
    headers = []
    
    for row in rows:
        headers = row.findall("tableheader")
        if len(headers) > 0:
            break
    return (rows, headers)

def summarize_table(table):
    table_summary = {}
    table_summary["headers"] = []
    rows, headers = _find_rows_and_headers(table)
    
    if len(headers) > 0:
        string = "Header: "
        for header in headers:
            table_summary["headers"].append(header.attrib["contents"])
            string = string + header.attrib["contents"] + " "
        #print(string.rstrip())
        
    for row in rows:
        cells = row.findall("tablecell")
        if len(cells) > 0:
            cell_name = cells[0].attrib["contents"]
            string = cell_name
            table_summary[cell_name] = []
            for cell in cells[1:]:
                table_summary[cell_name].append(cell.attrib["contents"])
                string = string + cell.attrib["contents"] + " " 
            #print(string.rstrip())
            
    return table_summary

def summarize_section(section):
    section_summary = {}
    section_summary["tables"] = []
    section_summary["subsections"] = {}
    
    #print("Section:", section.attrib["title"])
    tables = section.findall("table")
    sub_sections = section.findall("section")
    for table in tables:
        section_summary["tables"].append(summarize_table(table))
    #print("")
    for sub_section in sub_sections:
        section_summary["subsections"][sub_section.attrib["title"]] = summarize_section(sub_section)
    
    return section_summary

def power_xml_to_dict(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sections = root.findall("section")
    result = {}
    
    for section in sections:
        result[section.attrib["title"]] = summarize_section(section)
        
    return result

def start_test_batch_fast(results_path, project_path, run_target, pairs):
    # Prepare tcl script
    script = template_open.replace("$PROJ_PATH$", project_path)
    #script = script.replace("$PERIOD$", period)
    script = script.replace("$RUN$", run_target)
    for toggle_rate, static_prob in pairs:
        script = script + template_single_test
        script = script.replace("$TOGGLE_RATE$", str(toggle_rate))
        script = script.replace("$STATIC_PROB$", str(static_prob))
        #script = script.replace("$SWITCH_TARGET$", switch_target)
        script = script.replace("$REPORT_PATH$", results_path)
        script = script.replace("$REPORT_NAME$", f"{toggle_rate}_{static_prob}")
    with open(os.getcwd() + "/power_report.tcl", "w") as tcl_file:
        tcl_file.write(script)

    # Prepare bash script
    bash_script = os.getcwd()  + "/report_power.sh"
    with open(bash_script, "w") as script:
        script.write("#!/bin/bash \n")
        script.write(f"vivado -mode batch -source {os.getcwd()}/power_report.tcl\n")
    
    # Run script
    sub_proc = subprocess.Popen(["bash", bash_script])
    sub_proc.communicate()

    # Parse results
    for toggle_rate, static_prob in pairs:
        power_report_dict = power_xml_to_dict(f"{results_path}/{toggle_rate}_{static_prob}.xml")
        power_report_json = f"{results_path}/{toggle_rate}_{static_prob}.json"
        with open(power_report_json, "w") as json_file:
            json_file.write(json.dumps(power_report_dict, indent=2))

def sim_power_report(results_path, project_path, in_width, out_width, dtype_width, sim_duration_ns):
    # Prepare tcl script
    script = template_open.replace("$PROJ_PATH$", project_path)
    script = script.replace("$RUN$", "impl_1")
    script = script + template_sim_power
    script = script.replace("$TB_FILE_PATH$", os.getcwd() + "/switching_simulation_tb.v")
    script = script.replace("$SAIF_FILE_PATH$", os.getcwd() + "/switching.saif")
    script = script.replace("$SIM_DURATION_NS$", str(int(sim_duration_ns)))
    script = script.replace("$REPORT_PATH$", results_path)
    script = script.replace("$REPORT_NAME$", f"sim")
    with open(os.getcwd() + "/power_report.tcl", "w") as tcl_file:
        tcl_file.write(script)

    # Prepare testbench
    testbench = template_switching_simulation_tb.replace("$INSTREAM_WIDTH$", str(in_width))
    testbench = testbench.replace("$OUTSTREAM_WIDTH$", str(out_width))
    testbench = testbench.replace("$DTYPE_WIDTH$", str(dtype_width))
    testbench = testbench.replace("$RANDOM_FUNCTION$", "$urandom_range(0, {max})".format(max=2**dtype_width-1))
    with open(os.getcwd() + "/switching_simulation_tb.v", "w") as tb_file:
        tb_file.write(testbench)

    # Prepare shell script
    bash_script = os.getcwd()  + "/report_power.sh"
    with open(bash_script, "w") as script:
        script.write("#!/bin/bash \n")
        script.write(f"vivado -mode batch -source {os.getcwd()}/power_report.tcl\n")

    # Run script
    sub_proc = subprocess.Popen(["bash", bash_script])
    sub_proc.communicate()
    
    # Parse results
    power_report_dict = power_xml_to_dict(f"{results_path}/sim.xml")
    power_report_json = f"{results_path}/sim.json"
    with open(power_report_json, "w") as json_file:
        json_file.write(json.dumps(power_report_dict, indent=2))

def prepare_inputs(input_tensor, idt, wdt):
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert bipolar to binary
        return {"inp": (input_tensor + 1) / 2}
    else:
        return {"inp": input_tensor}

def bench_transformer(params, task_id, run_id, results_dir):

    output_dict = {}

    #create specialize_layers.json
    specialize_layers_dict = {
        "Defaults": {
            "preferred_impl_style": ["%s"%params["backend"], ["MVAU", "Thresholding"]]
        },
        "": {
            "preferred_impl_style": ""
        }
        }
    with open(os.path.join("", "specialize_layers_generated.json"), "w") as f:
        json.dump(specialize_layers_dict, f, indent=2)

    #create folding_config.json
    folding_dict = {
        "Defaults": {
            "ram_style": ["%s"%params["attention_ram_style"], ["ScaledDotProductAttention_hls"]],
            "mac_resource": ["%s"%params["attention_mac_resource"], ["ScaledDotProductAttention_hls"]],

            "mem_mode": ["%s"%params["mvau_hls_mem_mode"], ["MVAU_hls"]],
            "ram_style": ["%s"%params["mvau_hls_ram_style"], ["MVAU_hls"]],
            "resType": ["%s"%params["mvau_hls_resType"], ["MVAU_hls"]],

            "ram_style": ["%s"%params["mvau_rtl_ram_style"], ["MVAU_rtl"]],

            "mem_mode": ["%s"%params["mt_hls_mem_mode"], ["Thresholding_hls"]],
            "ram_style": ["%s"%params["mt_hls_ram_style"], ["Thresholding_hls"]],

            "depth_trigger_uram": ["%s"%params["mt_rtl_depth_trigger_uram"], ["Thresholding_rtl"]],
            "depth_trigger_bram": ["%s"%params["mt_rtl_depth_trigger_bram"], ["Thresholding_rtl"]],

            "impl_style": ["%s"%params["fifo_impl_style"], ["StreamingFIFO_rtl"]],
            "ram_style": ["%s"%params["fifo_ram_style"], ["StreamingFIFO_rtl"]],

            "ram_style": ["%s"%params["elementwise_ram_style"], ["ElementwiseAdd_hls"]]
        }
        }

    with open(os.path.join("", "folding_generated.json"), "w") as f:
        json.dump(folding_dict, f, indent=2)


    # Extract sequence length and embedding dimension from parameters
    seq_len, emb_dim = params["seq_len"], params["emb_dim"]
    # Create a configuration for building the scaled dot-product attention
    # operator to a hardware accelerator
    cfg = build_cfg.DataflowBuildConfig(
        # Unpack the build configuration parameters
        #**params["build"],

        # Directory to store the build outputs
        output_dir = "build",
        # Run synthesis to generate a .dcp for the stitched-IP output product
        stitched_ip_gen_dcp = False,
        # Target clock period, i.e., inverse of target frequency
        synth_clk_period_ns = 10.0,
        # Board to target with the build
        board = "RFSoC2x2",
        # Target shell flow = 'vivado_zynq' or 'vitis_alveo'
        shell_flow_type = "vivado_zynq",
        # Path to folding configuration file
        folding_config_file = "folding_generated.json",
        # Path to layer implementation style specialization config
        specialize_layers_config_file = "specialize_layers_generated.json",
        # Force the implementation of standalone thresholds to be able to use RTL
        # implementation of the MVU
        standalone_thresholds = True,
        # Maximum bit-width of quantizers converted to multi-thresholds
        max_multithreshold_bit_width = 8,
        # Maximum width of MVAU stream per PE
        mvau_wwidth_max = 36,
        #  # Optional: Start the build from a specific step
        #  start_step: "step_tidy_up_pre_attention"
        #  # Optional: Stop the build after a specific step
        #  stop_step: "step_hw_ipgen"
        # Metrics aggregation configuration


        # Print all warnings and compiler output to stdout
        verbose=True,
        # Generate and keep the intermediate outputs including reports
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],

        # Steps after which verification should be run
        #verify_steps=[
            # Verify the model after converting to the FINN onnx dialect
            #build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
            # Verify the model again using python mode after the default
            # streamlining step
            #build_cfg.VerificationStepType.STREAMLINED_PYTHON,
            # Verify the model again after tidy up transformations, right before
            # converting to HLS
            #build_cfg.VerificationStepType.TIDY_UP_PYTHON,
            # Verify the model after generating C++ HLS and applying folding
            #build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
        #],

        # File with test inputs for verification
        verify_input_npy="inp.npy",
        # File with expected test outputs for verification
        verify_expected_output_npy="out.npy",
        # Save the intermediate model graphs
        save_intermediate_models=True,
        # Avoid RTL simulation for setting the FIFO sizes
        auto_fifo_strategy=AutoFIFOSizingMethod.CHARACTERIZE,
        # Do not automatically set FIFO sizes as this requires RTL simulation
        # not implemented for the attention operator
        auto_fifo_depths=False,
        # Build steps to execute
        steps=[
            # Need to apply some tidy-up transformations before converting to
            # the finn dialect of onnx
            step_tidy_up_pre_attention,
            # Convert all QONNX Quant nodes to Multithreshold nodes
            "step_qonnx_to_finn",
            # Tidy up the graph after converting from QONNX to FINN format
            # Note: Triggers a verification step
            "step_tidy_up",
            # Positional encoding needs to be streamlined first with slightly
            # different order of certain streamlining transformations to avoid
            # weird rounding issue of intermediate results
            step_streamline_positional,
            # Custom streamlining for models containing attention operators
            step_streamline_attention,
            # Streamlining of the residual branches
            step_streamline_residual,
            # Streamline the normalization layers, i.e., transposed batch norm
            step_streamline_norms,
            # Another round using the default streamlining steps
            # Note: Triggers a verification step
            "step_streamline",
            # New conversion of the scaled dot-product attention pattern
            step_convert_attention_to_hw,
            # Another tidy-up step to remove unnecessary dimensions and
            # operations after converting the attention operators to HLS
            step_tidy_up_post_attention,
            # Convert the elementwise binary operations to hardware operators.
            # These include for example adding residual branches and positional
            # encoding
            step_convert_elementwise_binary_to_hw,
            # Properly replicate the stream feeding the query, key and value
            # projections
            step_replicate_streams,
            # Convert most other layers supported by FINN to HW operators
            "step_convert_to_hw",
            # Specialize HW layer implementations as either HLS or RTL
            "step_specialize_layers",
            "step_create_dataflow_partition",
            # Set the folding configuration to meet the cycles per sequence
            # target
            set_target_parallelization(seq_len, emb_dim),
            # Apply folding configuration, specifying hardware implementation
            # details
            # Note: This triggers a verification step
            step_apply_folding_config,
            "step_minimize_bit_width",
            # The ScaledDotProductAttention custom op does not define any
            # estimates
            "step_generate_estimate_reports",
            "step_hw_codegen",
            "step_hw_ipgen",
            # Set the attention- and residual-related FIFO depths insert FIFOs
            # and apply folding configuration once again
            set_fifo_depths(seq_len, emb_dim),
            # Run additional node-by-node verification in RTL simulation of the
            # model before creating the stitched IP
            # Note: end-to-end verification of the stitched IP in RTL simulation
            # is still not possible due to missing float IPs

            #node_by_node_cppsim,
            #node_by_node_rtlsim,

            "step_create_stitched_ip",
            # Attention does currently not support RTL simulation due to missing
            # float IPs.
            # "step_measure_rtlsim_performance",
            "step_out_of_context_synthesis",
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]
    )
    # Run the build process on the dummy attention operator graph
    start_time = time.time()
    build.build_dataflow_cfg("../finn_benchmarking/models/" + params["model_file"], cfg)
    output_dict["build_time"] = int(time.time() - start_time)
    

    report = "build/report/post_synth_resources.json"
    with open(report) as file:
        report_dict = json.load(file)
        output_dict["post_synth_res_full"] = report_dict

    with open(report) as file:
        # Load the JSON formatted report
        report = pd.read_json(file, orient="index")
    # Filter the reported rows according to some regex filter rule
    filter = "(top)"
    report = report.filter(regex=filter, axis="rows")
    # Generate a summary of the total resources
    summary = report.sum()
    output_dict["post_synth_res_top"] = summary.to_dict()

    

    return output_dict
