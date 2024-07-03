import itertools
import json
import os
import sys

def consolidate_logs(path):
    log = []
    i = 0
    while (i < 1024):
        if (os.path.isfile(os.path.join(path,"task_%d.json"%(i)))):
            with open(os.path.join(path,"task_%d.json"%(i)), "r") as f:
                log_task = json.load(f)
            log.extend(log_task)
        i = i + 1
    
    with open(os.path.join(path,"task_all.json"), "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    consolidate_logs(sys.argv[1])
