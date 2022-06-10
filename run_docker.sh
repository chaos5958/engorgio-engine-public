#!/bin/bash

sudo docker run -it \
        --shm-size=2gb \
        -p 5000:22 \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --gpus all \
        --rm \
        -v ${HOME}/docker/tensorrt:/workspace/research \
        --name tensorrt tensorrt /bin/bash
