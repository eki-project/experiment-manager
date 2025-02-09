import fcntl
import time
from datetime import datetime
import sys
import os
import re
import json
import errno
import pandas as pd
from fabric import Connection
from patchwork import transfers
import vxi11

def log(str):
    current_time = datetime.now()
    str_date_time = current_time.strftime("%d-%m_%H-%M: ")
    print(str_date_time + str)

def board_power_on():
    psu = vxi11.Instrument("131.234.250.74")
    psu.write(f":OUTP ON")

def board_power_off():
    psu = vxi11.Instrument("131.234.250.74")
    psu.write(f":OUTP OFF")

def board_power_measure():
    psu = vxi11.Instrument("131.234.250.74")
    return float(psu.ask(f":MEAS:POWE? CH1"))

if __name__ == '__main__':
    # triggered regularly as a service
    f = open ('lock', 'w')
    try: fcntl.lockf (f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError as e:
        if e.errno == errno.EAGAIN:
            sys.stderr.write ('[%s] Script already running.\n' % time.strftime ('%c') )
            sys.exit (-1)
        raise

    board_on = False
    # search for bitstreams on pfs job dir w/o measurement report
    for experiment in os.listdir("/mnt/pfs/hpc-prf-radioml/felix/jobs"):
        if os.path.isdir("/mnt/pfs/hpc-prf-radioml/felix/jobs/" + experiment + "/bitstreams"):
            for file in os.listdir("/mnt/pfs/hpc-prf-radioml/felix/jobs/" + experiment + "/bitstreams"):
                if re.match(r"run_\d*.bit", file):
                    if not os.path.isfile("/mnt/pfs/hpc-prf-radioml/felix/jobs/" + experiment + "/bitstreams/" + file.replace(".bit", "_load.xlsx")):
                        # un-measured bistream detected
                        log("Running power measurement for: %s/%s"%(experiment, file))
                        if not board_on:
                            log("Powering on board")
                            board_power_on()
                            board_on = True
                            log("Waiting for 120s")
                            time.sleep(120)

                        settings = {}
                        with open(os.path.join("/mnt/pfs/hpc-prf-radioml/felix/jobs/", experiment , "bitstreams/", file.replace(".bit", "_settings.json")), "r") as f:
                                settings = json.load(f)
                        log("Loaded settings json file")

                        # different frequencies?
                        # w/ and w/o external psu (due to instabilities)
                        experiment_flow = {
                            "experimentDesc" : "Automatically triggered flow",
                            "global" : {"FINN" : {"driver" : "/home/xilinx/power_measurement", 
                                                    "bitstream" : "/home/xilinx/power_measurement/experiments/%s/%s"%(experiment, file)},
                                        "PAF" : {
                                            "rails" : ["0V85", "1V2_PL", "1V8", "3V3", "2V5_DC", "1V2_PS", "1V1_DC", "3V5_DC", "5V0_DC"], "sensors" : [],
                                            #"power_supply_ip": "131.234.250.74",
                                            "board": "rfsoc2x2"
                                        },
                                        "experiment_path" : "/home/xilinx/power_measurement/harness_driver.py",
                                        "import_paths" : [],

                                        },
                            "experiments" : [
                                {
                                    "title" : "idle",
                                    "functions" : [
                                                {"name": "run_idle", "args" : [], 
                                                "kwargs" : {"frequency": settings["freq_mhz"]}
                                                }
                                    ],
                                    "num_runs" : 1,
                                    "warmup": 10,
                                },
                                {
                                    "title" : "load",
                                    "functions" : [
                                                {"name": "run_free", "args" : [], 
                                                "kwargs" : {"frequency": settings["freq_mhz"]}
                                                }
                                    ],
                                    "num_runs" : 1,
                                    "warmup": 10,
                                }
                            ]
                        }
                        with open("/mnt/pfs/hpc-prf-radioml/felix/jobs/" + experiment + "/bitstreams/" + file.replace(".bit", "_paf.json"), "w") as f:
                            json.dump(experiment_flow, f, indent=2)

                        log("Copying experiment flow cfg, bitstream, hwh to board")
                        source_path = "/mnt/pfs/hpc-prf-radioml/felix/jobs/" + experiment + "/bitstreams/"
                        target_path = "/home/xilinx/power_measurement/experiments/%s/"%(experiment)
                        prefix = file.replace(".bit", "")
                        with Connection("xilinx@ceg-307.cs.upb.de") as c:
                            c.run("mkdir -p " + target_path)
                            c.put(source_path + prefix + ".bit", target_path + prefix + ".bit")
                            c.put(source_path + prefix + ".hwh", target_path + prefix + ".hwh")
                            c.put(source_path + prefix + "_paf.json", target_path + prefix + "_paf.json")

                            log("Starting PAF")
                            c.run("sudo bash -c 'cd /home/xilinx/power_measurement && source /etc/profile.d/xrt_setup.sh && source /etc/profile.d/pynq_venv.sh && python paf/paframework/paframework.py {}'".format(target_path + prefix + "_paf.json"),
                                  pty=True)
                            
                            log("Saving output .xlsx files")
                            c.run("mv /home/xilinx/power_measurement/idle_run1.xlsx /home/xilinx/power_measurement/experiments/%s/%s_idle.xlsx"%(experiment, prefix))
                            c.run("mv /home/xilinx/power_measurement/load_run1.xlsx /home/xilinx/power_measurement/experiments/%s/%s_load.xlsx"%(experiment, prefix))
                            c.get(target_path + prefix + "_idle.xlsx", source_path + prefix + "_idle.xlsx")
                            c.get(target_path + prefix + "_load.xlsx", source_path + prefix + "_load.xlsx")

                        log("Parsing results and writing to compact log")
                        df = pd.read_excel(source_path + prefix + "_idle.xlsx")
                        power_pl_ps_idle = round(df["0V85_power"].mean() * 0.001, 3)
                        power_total_idle = round(df["total_power"].mean() * 0.001, 3)
                        df = pd.read_excel(source_path + prefix + "_load.xlsx")
                        power_pl_ps_load = round(df["0V85_power"].mean() * 0.001, 3)
                        power_total_load = round(df["total_power"].mean() * 0.001, 3)

                        # compute further power metrics from measurements
                        power_pl_dyn_load_comp = round(power_pl_ps_load - power_pl_ps_idle, 3)
                        power_total_dyn_load_comp = round(power_total_load - power_total_idle, 3)

                        power_log_path = "/mnt/pfs/hpc-prf-radioml/felix/jobs/" + experiment + "/power_measure.json"
                        if os.path.isfile(power_log_path):
                            with open(power_log_path, "r") as f:
                                logfile = json.load(f)
                        else:
                            logfile = []

                        logfile.append({
                            "run_id": int(prefix.replace("run_","")),
                            "output": {
                                "power": {
                                    "power_pl_ps_idle": power_pl_ps_idle,
                                    "power_total_idle": power_total_idle,
                                    "power_pl_ps_load": power_pl_ps_load,
                                    "power_total_load": power_total_load,
                                    "power_total_dyn_load_comp": power_total_dyn_load_comp,
                                    "power_pl_dyn_load_comp": power_pl_dyn_load_comp
                                }
                            }
                        })

                        with open(power_log_path, "w") as f:
                            json.dump(logfile, f, indent=2)

    if board_on:
        log("Powering board off after 30s wait")
        time.sleep(30)
        board_power_off()
        log("Done")
    else:
        log("No un-measured bitstreams found in experiment dirs")
