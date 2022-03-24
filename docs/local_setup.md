# Installation
## Setup a conda environment

```
git clone https://github.com/mees/vapo.git
cd vapo/
conda create -n vapo_env python=3.8
conda activate vapo_env
```
## Install pytorch

To install the voting layer the cudatoolkit installed with pytorch must match the native CUDA version (in /usr/local/cuda/) which will be used to compile the CUDA code. Otherwise, the compiled CUDA/C++ code may not be compatible with the conda-installed PyTorch.

First check your CUDA version with nvcc --version or in /usr/local/cuda/version.json then install [pytorch](https://pytorch.org/get-started/locally/) with the corresponding toolkit version. This code was tested with pytorch 1.11 and cuda 11.3.

```
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Install the Hough voting layer

To install the voting layer first install [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page).
```
git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror/
mkdir build/
cd build/
cmake ..
sudo make install
```

Go to the directory of the voting layer and run [setup.py](./vapo/affordance/hough_voting/setup.py). If you do not have sudo privileges, don't run `sudo make install` instead change the diretory in "include_dirs" to match where the eigen-git-mirror repo was downloaded, then run:

```
conda activate vapo
cd /VAPO_ROOT/vapo/affordance/hough_voting/
python setup.py install
```

## Install the VRENv
For more details refere to [VREnv setup](../VREnv/docs/setup.md)
```
cd /VAPO_ROOT/VREnv/
git clone https://github.com/bulletphysics/bullet3.git
cd bullet3

# Optional: patch bullet for selecting correct rendering device
# (only relevant when using EGL and multi-gpu training)
wget https://raw.githubusercontent.com/BlGene/bullet3/egl_remove_works/examples/OpenGLWindow/EGLOpenGLWindow.cpp -O examples/OpenGLWindow/EGLOpenGLWindow.cpp
pip install numpy
pip install -e .
```

## Finish the installation
```
cd /VAPO_ROOT/
pip install -e .
```
