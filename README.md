# VAPO
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/mees/vapo.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/mees/vapo/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/mees/vapo.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/mees/vapo/alerts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Affordance Learning from Play for Sample-Efficient Policy Learning**](https://arxiv.org/pdf/2203.00352.pdf)

Jessica Borja, [Oier Mees](https://www.oiermees.com/), [Gabriel Kalweit](https://nr.informatik.uni-freiburg.de/people/gabriel-kalweit), [Lukas Hermann](http://www2.informatik.uni-freiburg.de/~hermannl/), [Joschka Boedecker](https://nr.informatik.uni-freiburg.de/people/joschka-boedecker), [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

[ICRA 2022](https://www.icra2022.org/)




We present Visual Affordance-guided Policy Optimization (VAPO), a novel approach that extracts a selfsupervised visual affordance model from human teleoperated
play data and leverages it to enable efficient policy learning and motion planning. We combine model-based planning with model-free deep reinforcement learning (RL) to learn policies
that favor the same object regions favored by people, while requiring minimal robot interactions with the environment.
We find that our policies train 4x faster than the baselines and generalize better to novel objects because our visual affordance model can
anticipate their affordance regions. More information at our [project page](http://vapo.cs.uni-freiburg.de/).

<p align="center">
  <img src="http://vapo.cs.uni-freiburg.de/images/motivation.png" width="75%"/>
</p>

## Installation
Here we show how to install vapo on your local machine. Alternatively you can [install using Docker](./docs/docker_setup.md)

The installer is set to download pytorch 1.11 with cuda 11.3 by default. To correctly install the voting layer the cudatoolkit version installed with pytorch must match your systems CUDA version, which can be verified with nvcc --version command. Please modify the pytorch cudatoolkit of the environment accordingly. For mre details please refer to [local installation](./docs/local_setup.md)

```
git clone https://github.com/mees/vapo.git
cd vapo/
conda create -n vapo_env python=3.8
conda activate vapo_env
sh install.sh
```

### Install the Hough voting layer
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

## Quickstart
A quick tutorial on evaluating a pre-trained policy on the unseen dataset.

### Download the checkpoints
If you want to use the trained affordance models or policies, you can download them using the script in [trained_models](./trained_models/download_model_weights.sh)
```
  cd VAPO_ROOT/trained_models
  bash download_model_weights.sh
```
### Running an evaluation
We show how to run the evaluation for the policy on the unseen objets. For more details on running the policy please refer to [Policy](./docs/policy.md)

#### VAPO
The default configuration is set to run with the parameters of VAPO
```
python ./scripts/eval_tabletop.py scene=tabletop_random_unseen_15objs
```

#### Baseline
```
python ./scripts/eval_tabletop.py scene=tabletop_random_unseen_15objs test.folder_name=./trained_models/policy/tabletop/baseline
```
## Hardware Requirements
A single NVIDIA GPU with 8GB memory should be sufficient for training and evaluation. Altough the training time may vary.

Tested with:
GPU - ???
CPU - ???
RAM - ???
OS - Ubuntu 20.04.4

## Training

We show how to train a model from scratch assuming access to the playdata

1. Discover affordances in the playdata to create a dataset.
2. Train the affordance model for both the static and gripper camera.
3. Train the policy using the previously trained affordance models.
4. Evaluation.

### Dataset generation
Before training a model we first need to process the data to discover the affordances from the playdata. To do so we need to the [create_dataset.py](../scripts/create_dataset.py) script. 

If unspecified we assume the dataset will be stored in a folder named "datasets" in the VAPO_ROOT parent directory. This can be modified in [cfg_datacollection.yaml](../config/cfg_datacollection.yaml).

If you want to visualize the discovered affordances while is being created, add the flag viz=True

From VAPO_ROOT run:
```
python ./scripts/create_dataset.py play_data_dir=PLAY_DATA_DIR output_dir=DATA_DIR
```

This will create a dataset at DATA_DIR which can be used to train the affordance model.

### Training an affordance model
Here we show a small summary on how to train the affordance model. For a more detailed explanation on the available options please refer to the[affordance model documentation](./docs/affordance.md)

#### Set up wandb
We use wandb to log out results. By default it is set to offline under the configuration file [cfg_affordance](../config/cfg_affordance.yaml). Before training anything please log into your wandb and change the values of wandb_login in the previous file.

#### Train a model
The [training script](../scripts/train_affordance.py) uses the configuration defined in [cfg_affordance](../config/cfg_affordance.yaml). 

To train the affordance models you can run the following:

##### Gripper camera affordance model
```
python ./scripts/train_affordance.py model_name=aff_gripper dataset.cam=gripper dataset.data_dir=DATA_DIR
```

##### Static camera affordance model
```
python ./scripts/train_affordance.py model_name=aff_static dataset.cam=static dataset.data_dir=DATA_DIR
```

This will create an output at hydra_outputs/affordance_model/date/time. Alternatively you can specify were you want the model output by adding the flag `hydra.run.dir=CAM_AFF_OUT_FOLDER`


### Training a policy
After training the affordance model you can try training a policy and load the desired affordance model. All the saved affordance models, go to the hydra output directory. Here we load the model with the best mIoU as an example

#### VAPO
```
python ./scripts/train_tabletop.py \
gripper_cam_aff_path=GRIPPER_AFF_OUT_FOLDER/trained_models/best_val_miou.ckpt \
static_cam_aff_path=STATIC_AFF_OUT_FOLDER/trained_models/best_val_miou.ckpt \
```

#### Baseline
```
python ./scripts/train_tabletop.py \
gripper_cam_aff_path=GRIPPER_AFF_OUT_FOLDER/trained_models/best_val_miou.ckpt \
static_cam_aff_path=STATIC_AFF_OUT_FOLDER/trained_models/best_val_miou.ckpt \
affordance.gripper_cam.use_distance=False \
affordance.gripper_cam.use=False \
affordance.gripper_cam.densify_reward=False \
```

This will create an output to hydra_outputs/pickup/date/time. Alternatively you can define where to ouput the models using the flag hydra.run.dir=POLICY_TRAIN_FOLDER

## Evaluating your model
The evaluation script loads the parameters used for training by default. For this you need to provide the folder that is outputed as a result of training (POLICY_TRAIN_FOLDER). Additionally, we save different checkpoints. These can be found under POLICY_TRAIN_FOLDER/trained_models.

For this example we load the model with the most successful grasp on the all-objects evaluation

```
python ./scripts/eval_tabletop.py \
test.model_name=most_tasks_from_15 \
test.folder_name=POLICY_TRAIN_FOLDER
```

To test on a different scene you can add the flag scene=DESIRED_SCENE where DESIRED_SCENE is a yaml file name under [./config/scene](./config/scene/).

For instance to run the evaluation on the unseen objects run:
```
python ./scripts/eval_tabletop.py \
test.model_name=most_tasks_from_15 \
test.folder_name=POLICY_TRAIN_FOLDER \ 
scene=tabletop_random_unseen_15objs
```

For more details on using the policy please refer to [Policy](./docs/policy.md)

## Citation

If you find the dataset or code useful, please cite:

```
@inproceedings{borja22icra,
author = {Jessica Borja-Diaz and Oier Mees and Gabriel Kalweit and  Lukas Hermann and Joschka Boedecker and Wolfram Burgard},
title = {Affordance Learning from Play for Sample-Efficient Policy Learning},
booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation  (ICRA)},
year = 2022,
address = {Philadelphia, USA}
}
```

## License

MIT License
