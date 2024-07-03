from experiment_manager import system

cc9 = system(ssh_target = "cc9", hostname = "cc-9.cs.uni-paderborn.de", username = "fepaje", workdir = "/home/fepaje/jobs", use_slurm = False, ramdisk_path = None, environment={
    "FINN_XILINX_PATH": "/opt/Xilinx",
    "FINN_DOCKER_GPU": "NULL",
    "XILINXD_LICENSE_FILE": "27000@license5.uni-paderborn.de",
            })

# /scratch/hpc-prf-radioml/felix/jobs
# /scratch/hpc-prf-ekiapp/felix/jobs
# /scratch/hpc-prf-ekiapp/flash_only/felix/jobs
n2 = system(ssh_target = "n2_prox", hostname = "cc-9.cs.uni-paderborn.de", username = "fepaje", workdir = "/scratch/hpc-prf-radioml/felix/jobs", use_slurm = True, ramdisk_path = "/dev/shm", environment={
    "FINN_XILINX_PATH": "/opt/software/FPGA/Xilinx",
    "FINN_DOCKER_GPU": "NULL",
    "XILINXD_LICENSE_FILE": "27000@license5.uni-paderborn.de",
    "XILINX_LOCAL_USER_DATA": "no",
    "APPTAINER_CACHEDIR": "/scratch/hpc-prf-radioml/felix/APPTAINER_CACHE",
    "APPTAINER_TMPDIR": "/scratch/hpc-prf-radioml/felix/APPTAINER_TMP",
    "SINGULARITY_CACHEDIR": "/scratch/hpc-prf-radioml/felix/APPTAINER_CACHE",
    "SINGULARITY_TMPDIR": "/scratch/hpc-prf-radioml/felix/APPTAINER_TMP",
    "PYTHONUNBUFFERED": "TRUE",
    "LC_ALL": "C",
})
