#!/bin/bash

function _usage(){
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) -d [DIRECTORY]
EOF
}

[[ ($# -ge 1)  ]] || { _usage; exit 1;  }

while getopts ":d:" opt; do
    case $opt in
        d) directory="$OPTARG";;
        ?) exit 1;
    esac
done

if [ -z "${directory+x}" ]; then
    directory=${HOME}/docker/engorgio-pytorch
fi

sudo docker run -it \
        --shm-size=2gb \
        -p 5000:22 \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --gpus all \
        --rm \
        -v ${directory}:/workspace/research \
        --name engorgio_engine engorgio_engine /bin/bash
