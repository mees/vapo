# Dataset creation
Before training a model we first need to process the data to discover the affordances from the playdata. To do so we need to the [create_dataset.py](../scripts/create_dataset.py) script. 

If unspecified we assume the dataset will be stored in a folder named "datasets" in the VAPO_ROOT parent directory. 
This can be modified in [cfg_datacollection.yaml](../config/cfg_datacollection.yaml).

From `VAPO_ROOT` run:
```
python ./scripts/create_dataset.py play_data_dir=PLAY_DATA_DIR output_dir=DATA_DIR
```

If you want to visualize the affordances while is being created, add the flag `viz=True`:
```
python ./scripts/create_dataset.py play_data_dir=PLAY_DATA_DIR output_dir=DATA_DIR viz=True
```

For more information in the different parameters that can be changed, please refer to the configuration file [cfg_datacollection.yaml](../config/cfg_datacollection.yaml).

To visualize the transforms being applied to an image during training or validation, you cant test the [dataloader](../vapo/affordance/dataloader/datasets.py) by running:
```
python ./vapo/affordance/dataloader/datasets.py
```
Modify the main method of that same script to change between the `validation` and `training` dataloader. This will run the configuration on [cfg_affordance](../config/cfg_affordance.yaml). Therefore it takes as argument the same parameters that the training scrip.

# Training the affordance model
## Set up wandb
We use wandb to log out results. By default it is set to offline under the configuration file [cfg_affordance](../config/cfg_affordance.yaml). Before training anything please log into your wandb and change the values of wandb_login in the previous file.

## Train a model
The [training script](../scripts/train_affordance.py) uses the configuration defined in [cfg_affordance](../config/cfg_affordance.yaml). To train the affordance model you can try running the following:

```
python ./scripts/train_affordance.py model_name=aff_gripper dataset.cam=gripper dataset.data_dir=DATA_DIR
python ./scripts/train_affordance.py model_name=aff_static dataset.cam=static dataset.data_dir=DATA_DIR
```

where `DATA_DIR` points to a directory (relative or absolute) pointing to a dataset outputed by [create_dataset.py](./scripts/create_dataset.py). To get a better insight into what can be defined to trained the affordance model please refer to the configuration file [cfg_affordance](../config/cfg_affordance.yaml).


# Prediction visualization/ Inference
If desired you can specify a different configuration for the center prediction by modifying the parameters in model_cfg.hough_voting or specifying a different config in [./config/model_cfg/default.py](../config/aff_model/default.yaml).

Visualization configuration can be found in [viz_affordances.yaml](../config/viz_affordances.yaml). Please refer to this file for more information into the parameters it takes.
The visualization script can be found in [viz_affordances.py](../scripts/viz_affordances.py)

If the images come from a dataset created by create_dataset.py, you can specify the camera images for which you want to test with `cam_data=gripper` or `cam_data=static`. 
Otherwise set `cam_data=null` to load the default camera for the given model.

**Examples**:
```
python ./scripts/viz_affordances.py data_dir=DATA_DIR model_path=MODEL_HYDRA_OUTPUT_PATH
python ./scripts/viz_affordances.py data_dir=DATA_DIR model_path=MODEL_HYDRA_OUTPUT_PATH model_cfg=NEW_CFG_FILENAME
```
