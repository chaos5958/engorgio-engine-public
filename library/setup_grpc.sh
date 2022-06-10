#!/bin/bash

set -e

# packages
apt install -y build-essential autoconf libtool pkg-config

#grpc
pushd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local/grpc \
      ../..
make -j
make install
popd
