# Training
## VAPO
### Tabletop
### Drawer

## Baseline
### Tabletop
### Drawer

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