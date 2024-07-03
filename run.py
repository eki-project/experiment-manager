import sys

from experiment_manager import system, repository, directory, experiment
from systems import cc9, n2


def finn_build():

    dependencies = [
        # FINN
        repository(dir = "finn", remote = "https://github.com/fpjentzsch/finn.git", branch_commit = "feature/apptainer"), # branch or "tags/xyz" or commit hash
        #directory(path = "/home/felixj/WD/vitis_test/finn", exclude=["deps/*", "*.onnx"]),
    
        # RadioML
        #repository(dir = "finn-radioml", remote = "git@github.com:fpjentzsch/finn-radioml.git", branch_commit = "main"),
        directory(path = "/home/felixj/WD/vitis_test/finn-radioml", exclude=["finn_flow/output*"]),
    ]

    environment = {
        "FINN_SINGULARITY": "oras://ghcr.io/eki-project/experiment-manager/finn_apptainer_xilinx:72f3948",
        "FINN_XILINX_VERSION": "2022.2",
        "NUM_DEFAULT_WORKERS": "8",
    }

    slurm_options = {
        "--account": "hpc-prf-radioml", # hpc-prf-radioml OR hpc-prf-ekiapp
        #"--qos": "express", # test, express, fpgasynthesis (otherwise defaults to current project QoS (cont, lowcont, nocont))
        "--partition": "normal", # normal, largemem, hugemem, gpu, fpga
        "--time": "0-24", # [days-hours]
        "--nodes": "1",
        "--ntasks": "1",
        "--cpus-per-task": "8",
        "--mem": "64G", # memory per node
        "pre_commands": ["module reset", "module load system singularity", "ulimit -s unlimited"],
        # clean up artifacts that will be copied back to experiment dir:
        "post_commands": ["cd ..", "rm -rf finn/deps", "tar --use-compress-program='pigz -k ' -cf FINN_BUILD.tar.gz FINN_BUILD", "rm -rf FINN_BUILD"]
    }

    cmd = "export FINN_HOST_BUILD_DIR=$PWD/FINN_BUILD && cd finn && ./run-docker.sh build_custom ../finn-radioml/finn_flow"
    experiment(n2, tag="vgg10_test_cleanup", dependencies=dependencies, environment=environment, slurm_options=slurm_options, use_ramdisk=True, copy_back=True, cmd=cmd).launch()


def finn_benchmark_test():

    dependencies = [
        # FINN
        repository(dir = "finn", remote = "https://github.com/fpjentzsch/finn.git", branch_commit = "sample_parallelism_apptainer"), # branch or "tags/xyz" or commit hash
        #directory(path = "/home/felixj/WD/vitis_test/finn", exclude=["deps/*", "*.onnx"]),
    
        # Benchmarking scripts
        directory(path = "/home/felixj/WD/experiment-manager/finn_benchmarking"),
        directory(path = "/home/felixj/WD/experiment-manager/power_measurement"),
    ]

    environment = {
        "FINN_SINGULARITY": "oras://ghcr.io/eki-project/experiment-manager/finn_apptainer_xilinx:72f3948",
        "FINN_XILINX_VERSION": "2022.2",
        "NUM_DEFAULT_WORKERS": "4",
    }

    slurm_options = {
        "--account": "hpc-prf-radioml", # hpc-prf-radioml OR hpc-prf-ekiapp
        #"--qos": "express", # test, express, fpgasynthesis (otherwise defaults to current project QoS (cont, lowcont, nocont))
        "--partition": "normal", # normal, largemem, hugemem, gpu, fpga
        "--time": "0-24", # [days-hours]
        "--nodes": "1",
        "--ntasks": "1",
        "--cpus-per-task": "4",
        "--mem": "128G", # memory per node
        #"--array": "0-1", # %4 = run max. 4 jobs simultaneously
        "pre_commands": ["module reset", "module load system singularity"],
        "post_commands": []
    }

    cmd = "export FINN_HOST_BUILD_DIR=$PWD/FINN_BUILD && cd finn && ./run-docker.sh python ../finn_benchmarking/benchmarking.py mvau configs/mvau_test.json"
    experiment(n2, tag="mvau_sparsity_test", dependencies=dependencies, environment=environment, slurm_options=slurm_options, use_ramdisk=True, copy_back=True, cmd=cmd).launch()


def finn_benchmark():

    dependencies = [
        # FINN
        #repository(dir = "finn", remote = "https://github.com/fpjentzsch/finn.git", branch_commit = "feature/swg_reordering"), # branch or "tags/xyz" or commit hash
        directory(path = "/home/felixj/WD/finn", exclude=["deps/*", "*.onnx"]),
    
        # Benchmarking scripts
        directory(path = "/home/felixj/WD/experiment-manager/finn_benchmarking"),
        directory(path = "/home/felixj/WD/experiment-manager/power_measurement"),
    ]

    environment = {
        "FINN_SINGULARITY": "oras://ghcr.io/eki-project/experiment-manager/finn_apptainer_xilinx:72f3948",
        "FINN_XILINX_VERSION": "2022.2",
        "NUM_DEFAULT_WORKERS": "4",
    }

    slurm_options = {
        "--account": "hpc-prf-radioml", # hpc-prf-radioml OR hpc-prf-ekiapp
        #"--qos": "express", # test, express, fpgasynthesis (otherwise defaults to current project QoS (cont, lowcont, nocont))
        "--partition": "normal", # normal, largemem, hugemem, gpu, fpga
        "--time": "0-48", # [days-hours]
        "--nodes": "1",
        "--ntasks": "1",
        "--cpus-per-task": "4",
        "--mem": "64G", # memory per node
        #"--array": "0-63", # %4 = run max. 4 jobs simultaneously
        "pre_commands": ["module reset", "module load system singularity"],
        "post_commands": []
    }

    cmd = "export FINN_HOST_BUILD_DIR=$PWD/FINN_BUILD && cd finn && ./run-docker.sh python ../finn_benchmarking/benchmarking.py rtl_swg configs/rtl_swg_reordering.json"
    experiment(n2, tag="swg_reodering_fix", dependencies=dependencies, environment=environment, slurm_options=slurm_options, use_ramdisk=True, copy_back=True, cmd=cmd).launch()

def dummy_transformer():

    dependencies = [
        directory(path = "/home/felixj/WD/attention-test/finn"),
        #directory(path = "/home/felixj/WD/attention-dummy"),
    
        # Benchmarking scripts
        directory(path = "/home/felixj/WD/experiment-manager/finn_benchmarking"),
        directory(path = "/home/felixj/WD/experiment-manager/power_measurement"),
    ]

    environment = {
        "FINN_SINGULARITY": "/$PC2DATA/hpc-prf-ekiapp/FINN_IMAGES/xilinx/finn_dev.sif",
        "FINN_XILINX_VERSION": "2022.2",
        "NUM_DEFAULT_WORKERS": "4",
    }

    slurm_options = {
        "--account": "hpc-prf-radioml", # hpc-prf-radioml OR hpc-prf-ekiapp
        #"--qos": "express", # test, express, fpgasynthesis (otherwise defaults to current project QoS (cont, lowcont, nocont))
        "--partition": "normal", # normal, largemem, hugemem, gpu, fpga
        "--time": "0-24", # [days-hours]
        "--nodes": "1",
        "--ntasks": "1",
        "--cpus-per-task": "4",
        "--mem": "64G", # memory per node
        "--array": "0-35", # %4 = run max. 4 jobs simultaneously
        "pre_commands": ["module reset", "module load system singularity"],
        "post_commands": []
    }

    cmd = "export FINN_HOST_BUILD_DIR=$PWD/FINN_BUILD && cd finn && ./run-docker.sh python ../finn_benchmarking/benchmarking.py transformer configs/transformer_dummy_test.json"
    experiment(n2, tag="transformer_dummy_tiny", dependencies=dependencies, environment=environment, slurm_options=slurm_options, use_ramdisk=True, copy_back=False, cmd=cmd).launch()

def main():
    # parse arguments
    args = sys.argv[1:]

    #finn_build()
    #finn_benchmark_test()
    #finn_benchmark()

    dummy_transformer()

if __name__ == "__main__":
    main()
