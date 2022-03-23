# Dataset creation
Before training a model we first need to process the data to discover the affordances from the playdata. To do so we need to the [create_dataset.py](./scripts/create_dataset.py) script. For this we need to install additional libraries:
- Go to https://github.com/mees/VREnv and follow the installation instructions
-

` python ./scripts/create_dataset.py play_data_dir=PLAY_DATA_DIR output_dir=DATA_DIR`

# Training the affordance model
If you want to visualize how the data is labeled while is running add the flag viz=true.
Once the script is done we can start the training by running:

` python ./scripts/train_affordance.py model_name=test data_dir=DATA_DIR `

where DATA_DIR points to a directory (relative or absolute) on which the [create_dataset.py](./scripts/create_dataset.py) has created an output before.


# Prediction visualization/ Inference
If desired you can specify a different configuration for the center prediction by modifying the parameters in model_cfg.hough_voting or specifying a different config in [./config/model_cfg/default.py](./config/model_cfg/default.yaml).

Visualization configuration can be found in [viz_affordances.yaml](./config/viz_affordances.yaml)
Visualization script can be found in [viz_affordances.py](./scripts/viz_affordances.py)

If the images come from a dataset created by create_dataset.py, you can specify the camera images for which you want to test with cam_data=gripper or cam_data=static. Otherwise set cam_data=null

**Examples**:

` python ./scripts/viz_affordances.py data_dir=DATA_DIR model_path=MODEL_HYDRA_OUTPUT_PATH`

` python ./scripts/viz_affordances.py data_dir=DATA_DIR model_path=MODEL_HYDRA_OUTPUT_PATH model_cfg=NEW_CFG_FILENAME`
