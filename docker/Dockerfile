# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV HOME=/root
WORKDIR ${HOME}

########################### Install AWS dependencies ###########################

ARG AWS

# Check AWS argument is set
# AWS environment requires additional dependencies for its EFA network
RUN if [ -z "$AWS" ]; then echo "Pls indicate if the build is for AWS with --build-arg AWS=true/false"; exit 1; fi

# Copy install_aws_deps.sh in
COPY install_aws_deps.sh ${HOME}/install_aws_deps.sh

# Update apt
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-get update

# Install AWS dependencies if required
RUN if [ "$AWS" = "true" ]; then bash ${HOME}/install_aws_deps.sh; fi
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH

########################### Install DCP dependencies ###########################
# Install conda
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    ln -s ~/miniconda3/bin/conda /usr/bin/conda && \
    rm ~/miniconda3/miniconda.sh
# Remove existing torch installation
RUN pip uninstall torch -y
# Run all commands within a conda environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n dcp python=3.12
SHELL ["conda", "run", "-n", "dcp", "/bin/bash", "-c"]
# nvcr image comes with a constraint file that prevents installing other versions of pip packages
RUN rm /etc/pip/constraint.txt && touch /etc/pip/constraint.txt
RUN pip3 install --upgrade pip
RUN pip install --upgrade setuptools packaging
RUN conda install cmake ninja

# Build PyTorch.
# Note: we use a custom branch of PyTorch that supports all_to_all_single with
# zero byte send/recvs
RUN git clone --recursive -b alltoallv https://github.com/chenyu-jiang/pytorch.git ./pytorch
RUN cd pytorch && \
    pip install -r requirements.txt
RUN cd pytorch && \
    pip install mkl-static mkl-include && \
    conda install -c pytorch magma-cuda126 && \
    make triton
RUN cd pytorch && \
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}" && \
    pip install -e .

# Reset ld library path so the correct pytorch is used
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Install the custom FlashAttention (forked from version 2.6.3)
RUN git clone --recursive -b dcp https://github.com/chenyu-jiang/flash-attention.git ./flash-attention
RUN cd flash-attention && pip install -e . --no-build-isolation

# More dependencies
RUN pip install mtkahypar kahypar ortools transformers redis tiktoken datasets pybind11
RUN apt-get install -y redis
# Install orig version of flash-attn (used in baselines)
# Note: --no-dependencies is important to avoid reinstalling the latest version of torch
RUN pip install flash-attn==2.6.3 --no-build-isolation --no-dependencies
RUN mkdir -p ~/.ssh && echo "StrictHostKeyChecking no" >> ~/.ssh/config
RUN git clone https://github.com/chenyu-jiang/dcp.git && \
    cd dcp && \
    pip install -e . --no-build-isolation

# PaToH hypergraph partitioner
RUN wget https://faculty.cc.gatech.edu/~umit/PaToH/patoh-Linux-x86_64.tar.gz && \
    mkdir patoh && mv patoh-Linux-x86_64.tar.gz patoh && cd patoh && \
    tar -xzvf patoh-Linux-x86_64.tar.gz
ENV PATOH_PATH=${HOME}/patoh/build/Linux-x86_64

# pypatoh (Python bindings for PaToH)
RUN git clone https://github.com/chenyu-jiang/pypatoh.git && \
    cd pypatoh && \
    pip install -e . --no-build-isolation

########################### Install experiment dependencies and baselines ###########################
# transformer engine
# Note: this is a fork of the original Transformer Engine repo that supports
# masked CP
ENV CUDNN_INCLUDE_PATH=/usr/local/cuda/include
ENV CUDNN_PATH=/usr/local/cuda
ENV NVTE_FRAMEWORK=pytorch
RUN git clone --recursive -b stable https://github.com/chenyu-jiang/TransformerEngine.git && \
    cd TransformerEngine && \
    pip3 install -e . --no-build-isolation

# megatron-lm
RUN git clone https://github.com/NVIDIA/apex ./apex
RUN cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# use a fixed commit hash to avoid breaking changes
RUN git clone https://github.com/NVIDIA/Megatron-LM.git ./Megatron-LM
RUN cd Megatron-LM && git checkout 6185356290828c215a3b30c9df66a6adfb939f46 && pip install -e . --no-build-isolation --no-dependencies
# download GPT2 vocab file
RUN cd Megatron-LM && \
    mkdir vocabs && cd vocabs && \
    wget https://github.com/chenyu-jiang/Megatron-LM/raw/refs/heads/dynapipe/vocabs/gpt2-merges.txt && \
    wget https://github.com/chenyu-jiang/Megatron-LM/raw/refs/heads/dynapipe/vocabs/gpt2-vocab.json

# InternEvo (LoongTrain baseline)
RUN pip install git+https://github.com/InternLM/InternEvo.git --no-dependencies
RUN pip install pillow

# ring-flash-attn (baseline)
RUN pip install ring-flash-attn

# fixes numpy + pytorch import somehow
RUN conda install -c conda-forge libstdcxx-ng

########################### Install plotting tools ###########################
RUN pip install matplotlib seaborn

# set up bashrc
RUN conda init bash
RUN echo "conda activate dcp" >> $HOME/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/opt/amazon/efa/lib:/opt/amazon/openmpi5/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" >> $HOME/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:${HOME}/dcp:${HOME}/Megatron-LM" >> $HOME/.bashrc
RUN echo "export PATOH_PATH=${HOME}/patoh/build/Linux-x86_64" >> $HOME/.bashrc
SHELL ["/bin/bash"]
