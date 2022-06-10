#!/bin/bash

set -e

# packages
apt install -y nasm

# intel ipp
wget https://www.dropbox.com/s/k18tfw6yndhx99z/intel-ipp.zip? -O intel-ipp.zip
unzip intel-ipp.zip


# libjpeg-turbo
pushd libjpeg-turbo
mkdir -p build
pushd build
cmake -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_BUILD_TYPE=RELEASE  \
      -DENABLE_STATIC=FALSE       \
      -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-$LIBJPEG_TURBO_VER \
      -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
      -DCMAKE_ASM_NASM_COMPILER=/usr/bin/nasm \
      .. &&
make -j
make install
popd
rm -r build
popd
