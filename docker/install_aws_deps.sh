curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.37.0.tar.gz && \
    tar -xf aws-efa-installer-1.37.0.tar.gz && cd aws-efa-installer && \
    ./efa_installer.sh --skip-kmod -y

apt-get install libhwloc-dev -y

wget https://github.com/aws/aws-ofi-nccl/releases/download/v1.13.0-aws/aws-ofi-nccl-1.13.0.tar.gz && \
    tar -xf aws-ofi-nccl-1.13.0.tar.gz && cd aws-ofi-nccl-1.13.0 && \
    ./configure --prefix=/opt/aws-ofi-nccl --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws && \
    make -j && make install