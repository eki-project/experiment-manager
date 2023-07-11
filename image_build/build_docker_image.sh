#!/bin/bash

echo "Entering"

SCRIPT=$(readlink -f "$0")
echo "$SCRIPT"

cd ./finn
docker build -f docker/Dockerfile.finn --tag=finn_docker_export .
#docker save finn_docker_export -o finn_docker_image.tar
#todo: nocache
#todo: cleanup

echo "Exiting"
