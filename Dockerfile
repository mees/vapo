FROM nvidia/cuda:11.6.0-base-ubuntu20.04
WORKDIR /home/

# Install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda init bash

# Install cuda toolkit 11.3
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda

# Install conda environment
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN conda init bash
# RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo 'cd /home/' >> /root/.bashrc
RUN echo 'conda activate vapo_env' >> /root/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "vapo_env", "/bin/bash", "-c"]

# The code to run when container is started:
COPY start.sh .
RUN bash start.sh