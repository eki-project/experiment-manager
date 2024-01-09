# experiment-manager
Repository to collect scripts for launching and managing FINN-related synthesis or training jobs on Noctua 2 or other target machines.

Also contains the nightly Singularity image build for the official Xilinx/finn/dev branch:

[![ApptainerImage](https://github.com/eki-project/experiment-manager/actions/workflows/apptainer-image.yml/badge.svg)](https://github.com/eki-project/experiment-manager/actions/workflows/apptainer-image.yml)

Via the Github Actions page, this can also be manually triggered on public Github forks or custom branches/commits of FINN. The resulting Singularity images are pushed to the Github container registry of this repo and to the PC2DATA fileshare.
