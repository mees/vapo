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

First check your CUDA version with nvcc --version or in /usr/local/cuda/version.json then install [pytorch](https://pytorch.org/get-started/locally/) with the corresponding toolkit version. This code was tested with pytorch 1.10.1 and cuda 11.3.

```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## Install the VRENv
For more details refere to [VREnv setup](../VREnv/docs/setup.md)
```
pip install pybullet
cd /VAPO_ROOT/VREnv
pip install -e .
```

## Install VAPO
For more details refere to [VREnv setup](../VREnv/docs/setup.md)
```
cd /VAPO_ROOT/
pip install -e .
```

## Install the Hough voting layer

To install the voting layer first install [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page).
```
git clone https://gitlab.com/libeigen/eigen.git
cd eigen/
mkdir build/
cd build/
cmake ..
sudo make install
```

Go to the directory of the voting layer and run [setup.py](./vapo/affordance/hough_voting/setup.py). If you do not have sudo privileges, don't run `sudo make install` instead change the diretory in "include_dirs" to match where the eigen repo was downloaded, then run:

```
conda activate vapo_env
cd /VAPO_ROOT/vapo/affordance/hough_voting/
python setup.py install
```
