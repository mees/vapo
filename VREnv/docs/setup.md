# Setup
## Linux
Tested on Ubuntu 18.04 / 20.04 and Python 3.8
### Install SteamVR
- In terminal run `$ steam`, it will start downloading an update and create a `.steam` folder in your home directory.
- In Steam, create user account or use existing account.
- Click `Steam -> Settings -> Downloads -> Steam Library Folders -> Add Library Folder -> /media/hdd/SteamLibrary` to add existing installation of SteamVR to your Steam account
- Restart Steam
- Connect and turn on HTC VIVE
- Launch `Library -> SteamVR` (if not shown, check `[] Tools` box)
- If SteamVR throws an  `Error: setcap of vrcompositor-launcher failed`, run `$ sudo setcap CAP_SYS_NICE+ep /media/hdd/SteamLibrary/steamapps/common/SteamVR/bin/linux64/vrcompositor-launcher`
- Make sure Headset and controller are correctly detected, go through VR setup procedure
### Install Bullet
```
$ git clone https://github.com/bulletphysics/bullet3.git
$ cd bullet3

# Optional: patch bullet for selecting correct rendering device
# (only relevant when using EGL and multi-gpu training)
$ wget https://raw.githubusercontent.com/BlGene/bullet3/egl_remove_works/examples/OpenGLWindow/EGLOpenGLWindow.cpp -O examples/OpenGLWindow/EGLOpenGLWindow.cpp

# For building Bullet for VR  add -DUSE_OPENVR=ON to line 8 of build_cmake_pybullet_double.sh
# Run
$ ./build_cmake_pybullet_double.sh


$ pip install numpy  # important to have numpy installed before installing bullet
$ pip install -e .  # effectively this is building bullet a second time, but importing is easier when installing with pip

# add alias to your bashrc
alias bullet_vr="~/.steam/steam/ubuntu12_32/steam-runtime/run.sh </PATH/TO/BULLET/>bullet3/build_cmake/examples/SharedMemory/App_PhysicsServer_SharedMemory_VR"

```
### Install VREnv
```
$ cd VREnv
$ pip install -e .
```
### Run Bullet VR
- Close all open SteamVR windows.
- Run `$ ~/.steam/ubuntu12_32/steam-runtime/run.sh <PATH_CONTAINING_BULLET>/bullet3/build/examples/SharedMemory/App_PhysicsServer_SharedMemory_VR`
- You can add this command as an alias to your bashrc

### Speech output
```
$ sudo apt-get install espeak
$ pip install pyttsx3
```

## Windows
### Prerequisites
- python3 (https://www.python.org/downloads/)
- visual studio build tools for c++ (https://visualstudio.microsoft.com/de/downloads/)
- oculus rift software (https://www.oculus.com/setup/)
- streamvr (https://www.steamvr.com/de/)
- python modules numpy, numpy-quaternion, scipy, numba, pybullet

### Build Bullet
Necessary build of "App_PhysicsServer_SharedMemory_VR.exe"
- clone repository from https://github.com/bulletphysics/bullet3
  - checkout wanted/current version (sync with pybullet releases to prevent version mismatch on pybullet server connect)
- run build_visual_studio_vr_pybullet_double.bat
- build build3\vs2010\App_PhysicsServer_SharedMemory_VR.vcxproj as Release/win32
  - (conversion of vsproj version may be necessary)

Optionally build pybullet if versions mismatch
- build python bdist (via python setup.py sdist bdist_wheel)
- install package (e.g. pip install dist\pybullet*.whl)


### Install VREnv
 ```
cd VREnv
pip install -e .
```

### Run
- start bin\App_PhysicsServer_SharedMemory_VR_vs2010.exe
- run python <yourscript.py> with proper options


## Editor
    editor.py
### Prerequisites
- pybullet
- tkinter

### Run
- python editor.py

## Urdf single body creator
    urdfSingleBodyCreator.py
### Prerequisites
- meshlab_server for inertia, center of mass
- pybullet for v-hacd
- Only implemented for Linux
### Run
- python urdfSingleBodyCreator.py <directory>(required) --scale(optional) --mass(optional)
