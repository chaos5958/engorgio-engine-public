#!/bin/bash

docker build -t engorgio_engine:latest --build-arg ssh_prv_key="$(cat ~/.ssh/id_rsa)" --build-arg ssh_pub_key="$(cat ~/.ssh/id_rsa.pub)" .
