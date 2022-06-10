# Caution: do not push this image publicly since it contains a ssh key

# our local base image
FROM nvcr.io/nvidia/tensorrt:21.10-py3

LABEL description="Container for the SR-specialized engine" 
SHELL ["/bin/bash", "-c"]

# setting
WORKDIR /workspace/research

# install system packages 
ENV PATH /opt/conda/bin:$PATH
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        git mercurial subversion \
        g++ gdb make ninja-build rsync zip htop

# install cmake
RUN wget --quiet https://github.com/microsoft/CMake/releases/download/v3.19.4268486/cmake-3.19.4268486-MSVC_2-Linux-x86.sh
RUN chmod +x cmake-3.19.4268486-MSVC_2-Linux-x86.sh
RUN ./cmake-3.19.4268486-MSVC_2-Linux-x86.sh --skip-license --prefix=/usr

# install additional packages 
RUN apt-get install -y yasm nasm tmux htop

# install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs

ARG ssh_prv_key
ARG ssh_pub_key

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub

# Download ipp
RUN mkdir -p /workspace/library && \
    cd /workspace/library && \
    wget -q https://www.dropbox.com/s/k18tfw6yndhx99z/intel-ipp.zip? -O intel-ipp.zip && \
    unzip -q intel-ipp.zip

# Download libjpeg-turbo
RUN cd /workspace/library && \
    git clone https://github.com/libjpeg-turbo/libjpeg-turbo --branch 2.1.2 --single-branch 

# Build libjpeg-turbo
RUN cd /workspace/library/libjpeg-turbo && \
    mkdir -p build && \
    pushd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_BUILD_TYPE=RELEASE  \
      -DENABLE_STATIC=FALSE       \
      -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-$LIBJPEG_TURBO_VER \
      -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
      -DCMAKE_ASM_NASM_COMPILER=/usr/bin/nasm \
      .. && \
    make -j && \
    make install && \
    popd && \
    rm -r build 

# Download libvpx
RUN cd /workspace/library && \
    git clone https://github.com/kaist-ina/libvpx.git --branch engorgio --single-branch

# Build libvpx
RUN cd /workspace/library/libvpx && \
    ./configure_x86_library.sh && \
    make -j

# Download grpc
RUN apt install -y build-essential autoconf libtool pkg-config
RUN cd /workspace/library && \
    git clone --recurse-submodules https://github.com/grpc/grpc.git --branch v1.42.0 --single-branch

# Build grpc
RUN cd /workspace/library/grpc && \
	mkdir -p cmake/build && \
	cd cmake/build && \
	cmake -DgRPC_INSTALL=ON \
      	-DgRPC_BUILD_TESTS=OFF \
      	-DCMAKE_INSTALL_PREFIX=/usr/local/grpc \
      	../.. && \
	make -j && \
	make install

# Set environment variables
ENV LIBVPX_LIBRARY_PATH /workspace/library/libvpx
ENV IPP_LIBRARY_PATH /workspace/library/intel-ipp
ENV JPEG_LIBRARY_PATH /workspace/library/libjpeg-turbo
ENV LD_LIBRARY_PATH $IPP_LIBRARY_PATH/lib/intel64:$LD_LIBRARY_PATH
