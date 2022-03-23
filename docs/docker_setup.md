# Setup
## Install docker

Install [docker](https://docs.docker.com/engine/install/) by selecting the correct platform and following the instructions.

## Nvidia-docker

Install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable gpu usage.

For the full installation details on nvidia-docker, refer to the nvidia [documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

If you are using ubuntu, on the host computer setup the gpc keys.
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

Install the nvidia-docker2 package (and dependencies) after updating the package listing:
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

Restart the Docker daemon to complete the installation after setting the default runtime:
```
sudo systemctl restart docker
```

At this point, a working setup can be tested by running a base CUDA container:
```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Preparing the environment

### Build the image
In the directory where the Dockerfile is, run:
```
docker build -t vapo_image .
docker build -t test .
```

### Run a container

The following line will run the vapo_image under the docker container named "vapo_container". This will open a command line of an ubuntu filesystem in which you can run the code of this repo. 

The "--rm" flag removes the container after the session is closed.

The gpus from the host that are exposed to the container can be defined with the flag --gpus

The -it flag is to open an interactive session

```
docker run -it --gpus all --name vapo_container vapo_image bash
docker run -it --gpus all --rm --name test_container test_img bash
```
###  Resume the container
Start your container using container id: 
```
docker start vapo_container
```

Attach and run your container:
```
docker attach vapo_container
```

### Forwarding cv2 imshow to host

https://stackoverflow.com/questions/67099130/how-to-run-my-scrip-python-opencv-on-docker-ubuntu


# Docker commands
- See all images
```
    docker images
```

- See running containers
```
    docker ps
```

- See all containers
```
    docker ps -a
```

- Remove image
```
    docker image rm [IMG_ID or IMG_NAME]
```

- Remove container
```
    docker rm [CONTAINER_ID or CONTAINER_NAME]
```