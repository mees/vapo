FROM nvidia/cuda:11.3.0-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && \
    apt-get -y install sudo
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update
RUN apt-get install build-essential cmake -y
RUN apt-get install -y git
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN nvcc --version

# Add user
ENV user lg
RUN useradd -m -d /home/user user && \
    chown -R user /home/user && \
    adduser user sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER user
WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

# Install conda
ENV PATH="/home/user/miniconda3/bin:${PATH}"
ARG PATH="/home/user/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /home/user/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda --version

# Create the environment:
COPY --chown=user:user environment.yml .
RUN conda init bash
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo 'cd /home/user/' >> /home/user/.bashrc
RUN echo 'conda activate vapo_env' >> /home/user/.bashrc

# Install pybullet
WORKDIR /home/user/
RUN git clone https://github.com/bulletphysics/bullet3.git
RUN cd bullet3
RUN wget https://raw.githubusercontent.com/BlGene/bullet3/egl_remove_works/examples/OpenGLWindow/EGLOpenGLWindow.cpp -O /home/user/bullet3/examples/OpenGLWindow/EGLOpenGLWindow.cpp

SHELL ["conda", "run", "-n", "vapo_env", "/bin/bash", "-c"]
# Install pytorch
RUN conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install bullet
WORKDIR /home/user/bullet3
RUN pip install numpy
RUN pip install -e .

# Install VREnv
WORKDIR /home/user/
RUN git clone https://github.com/mees/vapo.git
WORKDIR /home/user/vapo/VREnv
RUN pip install -e .

# Install vapo
WORKDIR /home/user/vapo/
RUN pip install -e .

# Install hough voting layer
WORKDIR /home/user/
RUN git clone https://github.com/eigenteam/eigen-git-mirror.git

WORKDIR /home/user/eigen-git-mirror/
RUN mkdir build/

WORKDIR /home/user/eigen-git-mirror/build
RUN cmake ..
RUN sudo make install
WORKDIR /home/user/vapo/vapo/affordance/hough_voting/
RUN python setup.py install
