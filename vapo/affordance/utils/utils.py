import os

import hydra
from omegaconf import OmegaConf
from scipy.spatial.transform.rotation import Rotation as R
from torchvision import transforms

from vapo.affordance.affordance_model import AffordanceModel


def get_abs_path(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def euler_to_quat(euler_angles):
    """xyz euler angles to xyzw quat"""
    return R.from_euler("xyz", euler_angles).as_quat()


def quat_to_euler(quat):
    """xyz euler angles to xyzw quat"""
    return R.from_quat(quat).as_euler("xyz")


def get_transforms(transforms_cfg, img_size=None):
    transforms_lst = []
    transforms_config = transforms_cfg.copy()
    for cfg in transforms_config:
        if ("size" in cfg) and img_size is not None:
            cfg.size = img_size
        if "vapo.affordance_model.utils.transforms" in cfg._target_:
            cfg._target_ = cfg._target_.replace(
                "vapo.affordance_model.utils.transforms",
                "affordance.dataloader.transforms",
            )
        transforms_lst.append(hydra.utils.instantiate(cfg))

    return transforms.Compose(transforms_lst)


def load_from_hydra(cfg):
    # Initialize model
    model_path = get_abs_path(cfg.model_path)
    hydra_cfg_path = os.path.join(model_path, ".hydra/config.yaml")
    if os.path.exists(hydra_cfg_path):
        run_cfg = OmegaConf.load(hydra_cfg_path)
    else:
        print("path does not exist %s" % hydra_cfg_path)
        run_cfg = cfg
    model_cfg = run_cfg.model_cfg
    model_cfg.hough_voting = cfg.model_cfg.hough_voting

    # Load model
    checkpoint_path = os.path.join(model_path, "trained_models")
    checkpoint_path = os.path.join(checkpoint_path, cfg.model_name)
    if os.path.isfile(checkpoint_path):
        model = AffordanceModel.load_from_checkpoint(checkpoint_path, cfg=model_cfg).cuda()
        model.eval()
        print("model loaded")
    else:
        model = None
        print("No file found in: %s " % checkpoint_path)
    return model, run_cfg


def torch_to_numpy(x):
    return x.detach().cpu().numpy()
