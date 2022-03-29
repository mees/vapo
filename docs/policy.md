# Set up wandb
We use wandb to log out results. By default it is set to offline under the configuration file [the tabletop](../config/cfg_tabletop.yaml) or [drawer](../config/cfg_playtable.yaml). Before training anything please log into your wandb and change the values of wandb_login in the previous file.

# Training
The models and hydra configuration will be stored under VAPO_ROOT/task/hydra_outputs/date/time
If you want to log the outputs to wandb, you can add your credentials to the hydra configuration of either [the tabletop](../config/cfg_tabletop.yaml) or [drawer](../config/cfg_playtable.yaml) experiments. The default configuration is set to run with the parameters of VAPO.

## Tabletop

### VAPO
```
python ./scripts/train_tabletop.py
```

### Baseline
```
python ./scripts/train_tabletop.py affordance.gripper_cam.use_distance=False affordance.gripper_cam.use=False affordance.gripper_cam.densify_reward=False 
```
## Generalization
sending the scene parameter with the name of any of the yaml files in [./config/scene](../config/scene) will start the training with said configuration. For the generalization experiments we used the following:

### VAPO
The default configuration is set to run with the parameters of VAPO
```
python ./scripts/train_tabletop.py scene=tabletop_random_unseen_15objs
```

### Baseline
```
python ./scripts/train_tabletop.py model_name=baseline scene=tabletop_random_unseen_15objs affordance.gripper_cam.use_distance=False affordance.gripper_cam.use=False affordance.gripper_cam.densify_reward=False 
```

## Drawer

### VAPO
The default configuration is set to run with the parameters of VAPO
```
python ./scripts/train_playtable.py
```

### Baseline
```
python ./scripts/train_playtable.py model_name=baseline affordance.gripper_cam.use_distance=False affordance.gripper_cam.use=False affordance.gripper_cam.densify_reward=False 
```

# Testing
Adding the argument viz_obs=True will result in windows showing what the agent is seeing. To select the available scenes please see the list of the [scene hydra configuration files](../config/scene).
The camera configuration tabletop_render is the one used to produce the videos, which additionally to the robot observations includes one high resolution full-view render of the scene.

## Tabletop
From the root directory run:
### Vapo
```
    python ./scripts/test_tabletop.py viz_obs=true camera_conf=tabletop_render test.folder_name=./trained_models/policy/tabletop/vapo
```

## Baseline
```
    python ./scripts/test_tabletop.py viz_obs=true camera_conf=tabletop_render test.folder_name=./trained_models/policy/tabletop/baseline
```
