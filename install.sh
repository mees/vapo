#!/bin/bash
conda install pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
cd VREnv/
pip install -e .
cd ..
pip install -e .