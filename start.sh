# Install torch
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html -y

# Install hough voting layer
cd ~/
git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror/
mkdir build/
cd build/
cmake ..
make install
cd ~/vapo/vapo/affordance/hough_voting/
python setup.py install

# Install pybullet
cd ~/
git clone https://github.com/bulletphysics/bullet3.git
cd bullet3
wget https://raw.githubusercontent.com/BlGene/bullet3/egl_remove_works/examples/OpenGLWindow/EGLOpenGLWindow.cpp -O examples/OpenGLWindow/EGLOpenGLWindow.cpp
pip install numpy
pip install -e .

# Install vapo
git clone https://github.com/mees/vapo.git
cd ~/vapo/
pip install -e .
cd ./VREnv
pip install -e .
