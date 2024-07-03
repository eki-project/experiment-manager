#!/usr/bin/env python3

import os
import json
import xml.etree.ElementTree as ET
import subprocess
import numpy as np
import argparse

# pairs = [(5, 0.1), (5, 0.2), (5, 0.3), (5, 0.4), (5, 0.5), (5, 0.6), (5, 0.7), (5, 0.8), (5, 0.9), 
# (10, 0.1), (10, 0.2), (10, 0.3), (10, 0.4), (10, 0.5), (10, 0.6), (10, 0.7), (10, 0.8), (10, 0.9), 
# (15, 0.1), (15, 0.2), (15, 0.3), (15, 0.4), (15, 0.5), (15, 0.6), (15, 0.7), (15, 0.8), (15, 0.9), 
# (20, 0.1), (20, 0.2), (20, 0.3), (20, 0.4), (20, 0.5), (20, 0.6), (20, 0.7), (20, 0.8), (20, 0.9), 
# (25, 0.2), (25, 0.3), (25, 0.4), (25, 0.5), (25, 0.6), (25, 0.7), (25, 0.8), (30, 0.2), (30, 0.3), 
# (30, 0.4), (30, 0.5), (30, 0.6), (30, 0.7), (30, 0.8), (35, 0.2), (35, 0.3), (35, 0.4), (35, 0.5), 
# (35, 0.6), (35, 0.7), (35, 0.8), (40, 0.2), (40, 0.3), (40, 0.4), (40, 0.5), (40, 0.6), (40, 0.7), 
# (40, 0.8), (45, 0.3), (45, 0.4), (45, 0.5), (45, 0.6), (45, 0.7), (50, 0.3), (50, 0.4), (50, 0.5), 
# (50, 0.6), (50, 0.7), (55, 0.3), (55, 0.4), (55, 0.5), (55, 0.6), (55, 0.7), (60, 0.3), (60, 0.4), 
# (60, 0.5), (60, 0.6), (65, 0.4), (65, 0.5), (65, 0.6), (70, 0.4), (70, 0.5), (70, 0.6), (75, 0.4), 
# (75, 0.5), (75, 0.6), (80, 0.4), (80, 0.5), (85, 0.5), (90, 0.5), (95, 0.5), (100, 0.5)]

pairs = [(0, 0.0),(0, 0.1),(0, 0.2),(0, 0.3),(0, 0.4),(0, 0.5),(0, 0.6),(0, 0.7),(0, 0.8),(0, 0.9),(0, 1.0),
         (5, 0.1),(5, 0.2),(5, 0.3),(5, 0.4),(5, 0.5),(5, 0.6),(5, 0.7),(5, 0.8),(5, 0.9),
         (10, 0.1),(10, 0.2),(10, 0.3),(10, 0.4),(10, 0.5),(10, 0.6),(10, 0.7),(10, 0.8),(10, 0.9),
         (15, 0.1),(15, 0.2),(15, 0.3),(15, 0.4),(15, 0.5),(15, 0.6),(15, 0.7),(15, 0.8),(15, 0.9),
         (20, 0.1),(20, 0.2),(20, 0.3),(20, 0.4),(20, 0.5),(20, 0.6),(20, 0.7),(20, 0.8),(20, 0.9),
         (25, 0.2),(25, 0.3),(25, 0.4),(25, 0.5),(25, 0.6),(25, 0.7),(25, 0.8),
         (30, 0.2),(30, 0.3),(30, 0.4),(30, 0.5),(30, 0.6),(30, 0.7),(30, 0.8),
         (35, 0.2),(35, 0.3),(35, 0.4),(35, 0.5),(35, 0.6),(35, 0.7),(35, 0.8),
         (40, 0.2),(40, 0.3),(40, 0.4),(40, 0.5),(40, 0.6),(40, 0.7),(40, 0.8),
         (45, 0.3),(45, 0.4),(45, 0.5),(45, 0.6),(45, 0.7),
         (50, 0.3),(50, 0.4),(50, 0.5),(50, 0.6),(50, 0.7),
         (55, 0.3),(55, 0.4),(55, 0.5),(55, 0.6),(55, 0.7),
         (60, 0.3),(60, 0.4),(60, 0.5),(60, 0.6),
         (65, 0.4),(65, 0.5),(65, 0.6),
         (70, 0.4),(70, 0.5),(70, 0.6),
         (75, 0.4),(75, 0.5),(75, 0.6),
         (80, 0.4),(80, 0.5),
         (85, 0.5),
         (90, 0.5),
         (95, 0.5),
         (100, 0.5)]

project_path = ""
period = ""
run_target = ""

vivado_gen_post_synth_power_report_template = """
open_project  $PROJ_PATH$/finn_vivado_stitch_proj.xpr
open_run synth_1
create_clock -period 4 -name main_clk [get_ports -r .*_clk]
set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ $SWITCH_TARGET$
report_power -file $REPORT_PATH$/power_report.xml -format xml
"""

template_open = """
open_project  $PROJ_PATH$/finn_vivado_stitch_proj.xpr
open_run $RUN$
"""
#create clock not needed in template_open anymore 
#create_clock -period $PERIOD$ -name main_clk [get_ports -r .*_clk]


# TODO remove deassert resets after testing
template_single_test = """
set_switching_activity -toggle_rate $TOGGLE_RATE$ -static_probability $STATIC_PROB$ -hier -type $SWITCH_TARGET$ [get_cells -r finn_design_i/.*]
set_switching_activity -deassert_resets
report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
reset_switching_activity -hier -type $SWITCH_TARGET$ [get_cells -r finn_design_i/.*]
"""

template_junction_temp = """
set_operating_conditions -junction_temp $TEMP$
report_power -file $REPORT_PATH$/$REPORT_NAME$.xml -format xml
"""
def start_test_batch_fast(path, switch_target):
    results_path = os.getcwd() + "/results/" + path
    os.makedirs(results_path, exist_ok=True)

    script = template_open.replace("$PROJ_PATH$", project_path)
    script = script.replace("$PERIOD$", period)
    script = script.replace("$RUN$", run_target)


    for toggle_rate, static_prob in pairs:
        script = script + template_single_test
        script = script.replace("$TOGGLE_RATE$", str(toggle_rate))
        script = script.replace("$STATIC_PROB$", str(static_prob))
        script = script.replace("$SWITCH_TARGET$", switch_target)
        script = script.replace("$REPORT_PATH$", results_path)
        script = script.replace("$REPORT_NAME$", f"{toggle_rate}_{static_prob}")

    with open(results_path + "/power_report.tcl", "w") as tcl_file:
        tcl_file.write(script)

    bash_script = results_path + "/report_power.sh"
    with open(bash_script, "w") as script:
        script.write("#!/bin/bash \n")
        script.write(f"vivado -mode batch -source {results_path}/power_report.tcl\n")
    
    sub_proc = subprocess.Popen(["bash", bash_script])
    sub_proc.communicate()

    for toggle_rate, static_prob in pairs:
        power_report_dict = power_xml_to_dict(f"{results_path}/{toggle_rate}_{static_prob}.xml")
        power_report_json = f"{results_path}/{toggle_rate}_{static_prob}.json"

        with open(power_report_json, "w") as json_file:
            json_file.write(json.dumps(power_report_dict, indent=4))


def junction_test():
    results_path = os.getcwd() + "/results_temp/"
    os.makedirs(results_path, exist_ok=True)

    script = template_open.replace("$PROJ_PATH$", project_path)
    script = script.replace("$PERIOD$", period)
    script = script.replace("$RUN$", run_target)

    for i in range(30, 130, 10):
        script = script + template_junction_temp
        script = script.replace("$TEMP$", str(i))
        script = script.replace("$REPORT_PATH$", results_path)
        script = script.replace("$REPORT_NAME$", str(i))

    with open(results_path + "/temp_report.tcl", "w") as tcl_file:
        tcl_file.write(script)

    bash_script = results_path + "/temp_report.sh"
    with open(bash_script, "w") as script:
        script.write("#!/bin/bash \n")
        script.write(f"vivado -mode batch -source {results_path}/temp_report.tcl\n")
    
    sub_proc = subprocess.Popen(["bash", bash_script])
    sub_proc.communicate()

    for i in range(30, 130, 10):
        temp_report_dict = power_xml_to_dict(f"{results_path}/{i}.xml")
        temp_report_json = f"{results_path}/{i}.json"

        with open(temp_report_json, "w") as json_file:
            json_file.write(json.dumps(temp_report_dict, indent=4))

def test():
    script = template_open + template_single_test
    script = script.replace("$PROJ_PATH$", project_path)

    #setup test 1
    script = script.replace("$TOGGLE_RATE$", "100")
    script = script.replace("$STATIC_PROB$", "0.5")
    script = script.replace("$SWITCH_TARGET$", "-hier -type lut [get_cells -r finn_design_i/.*]")
    script = script.replace("$REPORT_PATH$", ".")
    script = script.replace("$REPORT_NAME$", "100_0.5")
    script = script.replace("$RESET_TARGET$", "lut [get_cells -r finn_design_i/.*]")

    #setup test 2
    script = script + template_single_test
    script = script.replace("$TOGGLE_RATE$", "25")
    script = script.replace("$STATIC_PROB$", "0.5")
    script = script.replace("$SWITCH_TARGET$", "-hier -type lut [get_cells -r finn_design_i/.*]")
    script = script.replace("$REPORT_PATH$", ".")
    script = script.replace("$REPORT_NAME$", "25_0.5")
    script = script.replace("$RESET_TARGET$", "lut [get_cells -r finn_design_i/.*]")

    with open("power_report.tcl", "w") as tcl_file:
        tcl_file.write(script)



def start_test_batch(path, tcl_script):
    results_path = os.getcwd() + "/results/" + path
    os.makedirs(results_path, exist_ok=True)

    for toggle_rate in range(5, 105, 5):
        static_probs = [np.around(x, decimals=1) for x in np.arange(0.0, 1.1, 0.1) if x >= toggle_rate/200 and x <= 1 - toggle_rate/200]

        for static_prob in static_probs:
            report_path = os.path.abspath(results_path + "/" + str(toggle_rate) + "_" + str(static_prob))
            os.makedirs(report_path, exist_ok=True)

            run_script = tcl_script.replace("$PROJ_PATH$", project_path)
            run_script = run_script.replace("$REPORT_PATH$", report_path)
            run_script = run_script.replace("$TOGGLE_RATE$", str(toggle_rate))
            run_script = run_script.replace("$STATIC_PROB$", str(static_prob))
            
            with open(report_path + "/power_report.tcl", "w") as tcl_file:
                tcl_file.write(run_script)

            bash_script = report_path + "/report_power.sh"
            with open(bash_script, "w") as script:
                script.write("#!/bin/bash \n")
                script.write(f"vivado -mode batch -source {report_path}/power_report.tcl\n")
            
            sub_proc = subprocess.Popen(["bash", bash_script])
            sub_proc.communicate()

            power_report_dict = power_xml_to_dict(report_path + "/power_report.xml")
            power_report_json = report_path + "/power_report.json"
            with open(power_report_json, "w") as json_file:
                json_file.write(json.dumps(power_report_dict, indent=4))


def input_only():
    print("STARTING POWER REPORT FOR INPUT DATA")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "[get_nets -r finn_design_i/s_axis_0_tdata.*]")
    start_test_batch("input_only", tcl_script)

def io_output():
    print("STARTING POWER REPRT FOR IO_OUTPUT")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type io_output [get_cells -r finn_design_i/.*]")
    start_test_batch("io_output", tcl_script)

def io_bidir_enable():
    print("STARTING POWER REPRT FOR IO_BIDIR_ENABLE")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type io_bidir_enable [get_cells -r finn_design_i/.*]")
    start_test_batch("io_bidir_enable", tcl_script)

def register():
    print("STARTING POWER REPRT FOR REGISTER")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type register [get_cells -r finn_design_i/.*]")
    start_test_batch("register", tcl_script)

def lut_ram():
    print("STARTING POWER REPRT FOR LUT_RAM")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type lut_ram [get_cells -r finn_design_i/.*]")
    start_test_batch("lut_ram", tcl_script)

def lut():
    print("STARTING POWER REPRT FOR LUT")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type lut [get_cells -r finn_design_i/.*]")
    start_test_batch("lut", tcl_script)

def dsp():
    print("STARTING POWER REPRT FOR DSP")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type dsp [get_cells -r finn_design_i/.*]")
    start_test_batch("dsp", tcl_script)

def bram_enable():
    print("STARTING POWER REPRT FOR BRAM_ENABLE")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type bram_enable [get_cells -r finn_design_i/.*]")
    start_test_batch("bram_enable", tcl_script)

def bram_wr_enable():
    print("STARTING POWER REPRT FOR BRAM_WR_ENABLE")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type bram_wr_enable [get_cells -r finn_design_i/.*]")
    start_test_batch("bram_wr_enable", tcl_script)

def gt_txdata():
    print("STARTING POWER REPRT FOR GT_TXDATA")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type gt_txdata [get_cells -r finn_design_i/.*]")
    start_test_batch("gt_txdata", tcl_script)

def gt_rxdata():
    print("STARTING POWER REPRT FOR GT_RXDATA")
    tcl_script = vivado_gen_post_synth_power_report_template.replace("$SWITCH_TARGET$", "-hier -type gt_rxdata [get_cells -r finn_design_i/.*]")
    start_test_batch("gt_rxdata", tcl_script)

def run(args):
    if(args.all):
        start_test_batch_fast("io_output", "io_output")
        start_test_batch_fast("io_bidir_enable", "io_bidir_enable")
        start_test_batch_fast("register", "register")
        start_test_batch_fast("lut_ram", "lut_ram")
        start_test_batch_fast("lut", "lut")
        start_test_batch_fast("dsp", "dsp")
        start_test_batch_fast("bram_enable", "bram_enable")
        start_test_batch_fast("bram_wr_enable", "bram_wr_enable")
        start_test_batch_fast("gt_txdata", "gt_txdata")
        start_test_batch_fast("gt_rxdata", "gt_rxdata")
    elif(args.input_only):
        input_only()
    elif(args.io_output):
        io_output()
    elif(args.lut):
        start_test_batch_fast("lut", "lut")
    elif(args.temp):
        print("temperature test called")
        junction_test()
    # elif(args.test):
    #     print(f"proj-path: {project_path}")
    #     print(f"period: {args.period}, type {type(args.period)}")
    

if __name__ == "__main__":
    print("called from command line")
    parser = argparse.ArgumentParser(
                    prog='Power Report Sweep Tool',
                    description='Helps generating a sweep of vivado power reports with various settings',
                    epilog='')
    
    parser.add_argument('-a', '--all', action="store_true", help="Generate power sweep for all available settings.")
    parser.add_argument('-i', '--input-only', action="store_true", help="Only vary settings for input.")
    parser.add_argument('-io1', '--io-output', action="store_true", help="Only vary settings for io_output.")
    parser.add_argument('-io2', '--io-bidir-enable', action="store_true", help="Only vary settings for io_bidir_enable.")
    parser.add_argument('-reg', '--register', action="store_true", help="Only vary settings for register.")
    parser.add_argument('-lr', '--lut-ram', action="store_true", help="Only vary settings for lut_ram.")
    parser.add_argument('-l', '--lut', action="store_true", help="Only vary settings for lut.")
    parser.add_argument('-d', '--dsp', action="store_true", help="Only vary settings for dsp.")
    parser.add_argument('-b', '--bram-enable', action="store_true", help="Only vary settings for bram_enable.")
    parser.add_argument('-bw', '--bram-wr-enable', action="store_true", help="Only vary settings for bram_wr_enable.")
    parser.add_argument('-gtx', '--gt-txdata', action="store_true", help="Only vary settings for gt_txdata.")
    parser.add_argument('-grx', '--gt-rxdata', action="store_true", help="Only vary settings for gt_rxdata.")
    # parser.add_argument('-t', '--test', action="store_true", help="TEST FUNCTION")
    parser.add_argument('-tmp', '--temp', action="store_true", help="Perform junction temperature test.")
    parser.add_argument('-path', '--proj-path', action="store", help="Specify the path to the vivado project", required=True)
    parser.add_argument('-per', '--period', action="store", help="Specify the synthesis period used in ns", required=True)
    parser.add_argument('-r', '--run', action="store", help="Specify the run in which the reports are to be generated (e.g. synth_1, impl_1).", required=True)

    args = parser.parse_args()
    project_path = os.path.abspath(args.proj_path)
    period = args.period
    run_target = args.run 
    
    run(args)