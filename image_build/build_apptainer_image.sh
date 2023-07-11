#!/bin/bash

echo "Hello World"

SCRIPT=$(readlink -f "$0")
echo "$SCRIPT"

apptainer build finn_singularity_image.sif docker-daemon://finn_docker_export:latest

#todo: cleanup

echo "Exiting"