import itertools
import json
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold

from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor
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
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.replace_verilog_relpaths import ReplaceVerilogRelPaths
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.synth_ooc import SynthOutOfContext


def _infer_sparse_weight_tensor(W_conv, k_h, k_w, channels):
    W_sparse = np.zeros((channels, channels, k_h, k_w), dtype=np.float32)
    for ch in range(channels):
        W_sparse[ch][ch] = W_conv[ch][0]
    W_conv = W_sparse.astype(np.float32)
    W_matmul = W_conv.transpose(0, 2, 3, 1)
    W_matmul = W_matmul.reshape(channels, channels * k_h * k_w)
    W_matmul = W_matmul.T

    return W_matmul


def _calculate_dot_prod_range(dt_a, dt_b, len):
    """Returns the (min,max) values a dot product between two (un)signed vectors of
    types dt_a and dt_b of len elements can take."""
    min_prod = 2**30
    max_prod = -(2**30)
    for a_val in [dt_a.min(), dt_a.max()]:
        for b_val in [dt_b.min(), dt_b.max()]:
            prod = a_val * b_val * len
            if prod < min_prod:
                min_prod = prod
            if prod > max_prod:
                max_prod = prod
    return (min_prod, max_prod)


def _make_single_vvau_modelwrapper(
    W,
    pe,
    simd,
    k_h,
    k_w,
    channels,
    dim_h,
    dim_w,
    wdt,
    idt,
    odt,
    T=None,
    tdt=None,
    mem_mode="const",
    ram_style="auto"
):
    in_shape = [1, dim_h, dim_w, k_h * k_w * channels]  # [N, H, W, K*K*CH]
    out_shape = [
        1,
        dim_h,
        dim_w,
        channels,
    ]  # [N, H, W, OFM_CH] (OFM_CH=IFM_CH because depthwise convolution)

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    if T is not None:
        no_act = 0
        node_inp_list = ["inp", "weights", "thresh"]
        actval = odt.min()
    else:
        no_act = 1
        node_inp_list = ["inp", "weights"]
        actval = 0

    VVAU_node = helper.make_node(
        "VectorVectorActivation",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        PE=pe,
        SIMD=simd,
        Dim=[dim_h, dim_w],
        Channels=channels,
        Kernel=[k_h, k_w],
        resType="lut",
        ActVal=actval,
        inputDataType=idt.name,
        weightDataType=wdt.name,
        outputDataType=odt.name,
        noActivation=no_act,
        mem_mode=mem_mode,
        ram_style=ram_style
    )

    graph = helper.make_graph(
        nodes=[VVAU_node], name="vvau_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="vvau-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)
    model.set_tensor_datatype("weights", wdt)

    model.set_initializer("weights", W)
    model.set_tensor_shape("weights", (channels, 1, k_h, k_w))

    if T is not None:
        model.set_tensor_datatype("thresh", tdt)
        model.set_initializer("thresh", T)

    # Minimize accumulator width to obtain realistic HLS reports
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    return model


def prepare_inputs(input_tensor):
    return {"inp": input_tensor}


def test_fpgadataflow_vvau(
    idt, wdt, act, pe, simd, dim_h, dim_w, k_h, k_w, channels, mem_mode, ram_style, only_estimates, skip_rtlsim, skip_synth
):
    if pe == "channels":
        pe = channels
    if pe == "channels/2":
        pe = channels // 2

    if wdt == "idt":
        wdt = idt
    if act == "idt":
        act = idt

    if dim_w == 1 and k_w != 1:
        return

    if channels % pe != 0:
        return

    #if pe < channels and simd > 1:
    #    pytest.skip("Do not apply SIMD parallelism before max PE parallelism")

    # skip SIMD unfolding before PE unfolding, except for PE=1 case (for experimentation)
    if simd > 1 and pe != 1 and pe != channels:
        return

    # Generate weights in expected shape for ONNX and HLS node
    W = gen_finn_dt_tensor(wdt, (channels, 1, k_h, k_w))  # shape: [channels, 1, k, k]
    W_onnx = _infer_sparse_weight_tensor(
        W, k_h, k_w, channels
    )  # shape: [k*k*channels, channels]

    # Generate inputs in expected format for ONNX and HLS node
    x = gen_finn_dt_tensor(idt, (1, dim_h, dim_w, k_h * k_w * channels))
    x_vvau = x.reshape(1, dim_h, dim_w, k_h * k_w, channels // pe, pe)
    x_vvau = x_vvau.transpose(0, 1, 2, 4, 3, 5)
    x_vvau = x_vvau.reshape(1, dim_h, dim_w, channels * k_h * k_w)

    if act is None:
        T = None
        tdt = None
        odt = DataType["INT32"]
    else:
        odt = act
        (min_v, max_v) = _calculate_dot_prod_range(idt, wdt, k_h * k_w ) #### *channel !!DEBUG
        n_steps = act.get_num_possible_values() - 1
        T = np.random.randint(min_v, max_v - 1, (channels, n_steps)).astype(np.float32)
        T = np.sort(T, axis=1)
        tdt = DataType["INT32"]

    model = _make_single_vvau_modelwrapper(
        W, pe, simd, k_h, k_w, channels, dim_h, dim_w, wdt, idt, odt, T, tdt, mem_mode, ram_style
    )

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(GiveUniqueNodeNames())
    if not only_estimates:
        model = model.transform(PrepareIP("xczu7ev-ffvc1156-2-e", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    node = model.get_nodes_by_op_type("VectorVectorActivation")[0]
    inst = getCustomOp(node)

    output_dict = {
                "idt": str(idt),
                "wdt": str(wdt),
                "act": str(act),
                "pe": pe, 
                "simd": simd, 
                "dim_h": dim_h,
                "dim_w": dim_w,
                "k_h": k_h,
                "k_w": k_w,
                "channels": channels,
                "mem_mode": mem_mode
                }

    exp_cycles_dict = model.analysis(exp_cycles_per_layer)
    exp_cycles = exp_cycles_dict[node.name]
    exp_res_dict = model.analysis(res_estimation)
    exp_res = exp_res_dict[node.name]
    exp_res_hls_dict = model.analysis(hls_synth_res_estimation)
    exp_res_hls = exp_res_hls_dict[node.name]

    output_dict["Est. Cycles"] = exp_cycles
    output_dict["Est. LUT"] = exp_res["LUT"]
    output_dict["Est. BRAM"] = exp_res["BRAM_18K"] * 0.5
    output_dict["Est. URAM"] = exp_res["URAM"]
    output_dict["HLS est. LUT"] = exp_res_hls["LUT"]
    output_dict["HLS est. BRAM"] = int(exp_res_hls["BRAM_18K"]) * 0.5
    output_dict["HLS est. URAM"] = exp_res_hls["URAM"]

    if only_estimates:
        return output_dict
    
    if not skip_rtlsim:
        input_dict = prepare_inputs(x_vvau)
        print("Performing RTL SIM..")
        oxe.execute_onnx(model, input_dict)["outp"]

        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        output_dict["Cycles"] = cycles_rtlsim
        print("RTLSIM cycles: %d"%cycles_rtlsim)

    if not skip_synth:
        print("Synthesizing..")
        model = model.transform(ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP("xczu7ev-ffvc1156-2-e", 5))
        model = model.transform(SynthOutOfContext(part="xczu7ev-ffvc1156-2-e", clk_period_ns=5))
        ooc_res_dict = eval(model.get_metadata_prop("res_total_ooc_synth"))
        output_dict["LUT"] = ooc_res_dict["LUT"]
        output_dict["BRAM"] = ooc_res_dict["BRAM_18K"] * 0.5 + ooc_res_dict["BRAM_36K"]
        output_dict["URAM"] = ooc_res_dict["URAM"]
        output_dict["WNS"] = ooc_res_dict["WNS"]
        output_dict["Fmax"] = ooc_res_dict["fmax_mhz"]

    return output_dict

