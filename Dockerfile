# our local base image
FROM nvcr.io/nvidia/tensorrt:20.12-py3

LABEL description="Container for use with Visual Studio" 
SHELL ["/bin/bash", "-c"]

# install anaconda
ENV PATH /opt/conda/bin:$PATH
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
        libglib2.0-0 libxext6 libsm6 libxrender1 \
        git mercurial subversion
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        conda update conda && \
        conda update anaconda && \
        conda update --all && \
        echo "conda activate base" >> ~/.bashrc

# install packages for VS remote build
RUN apt-get install -y g++ gdb make ninja-build rsync zip

# install openssh
RUN apt-get update && apt-get install -y g++ rsync zip openssh-server make git
RUN mkdir -p /var/run/sshd
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \ 
   ssh-keygen -A 

# install turbojpeg
RUN git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git 
RUN pushd libjpeg-turbo && cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local && make -j16 && make install && popd
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# install cmake
RUN wget --quiet https://github.com/microsoft/CMake/releases/download/v3.19.4268486/cmake-3.19.4268486-MSVC_2-Linux-x86.sh
RUN chmod +x cmake-3.19.4268486-MSVC_2-Linux-x86.sh
RUN ./cmake-3.19.4268486-MSVC_2-Linux-x86.sh --skip-license --prefix=/usr

# install vcpkg
RUN git clone https://github.com/microsoft/vcpkg && \
    cd vcpkg && \
    ./bootstrap-vcpkg.sh 

# install C++ packages
RUN cd vcpkg && \
    ./vcpkg install cxxopts

# Setting the root account
RUN echo "root:ina8024@@" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

# expose port 22 
EXPOSE 22

# workspace
WORKDIR /workspace/research

# start ssh
ENTRYPOINT service ssh restart && /bin/bash
