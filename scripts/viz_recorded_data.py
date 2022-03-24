import os

import cv2
import hydra
import numpy as np
import tqdm

from vapo.affordance.dataset_creation.core.utils import get_files
from vapo.affordance.utils.utils import get_abs_path


def normalizeImg(low, high, img):
    imgClip = np.clip(img, low, high)
    maxVal = np.max(imgClip)
    minVal = np.min(imgClip)
    return np.uint8((255.0) / (maxVal - minVal) * (imgClip - maxVal) + 255.0)


def viz_data(cfg):
    # Episodes info
    # Simulation
    data_dir = get_abs_path(cfg.play_data_dir)
    files = get_files(data_dir, "npz", recursive=True)  # Sorted files
    if not cfg.labeling.real_world:
        ep_start_end_ids = np.load(os.path.join(data_dir, "ep_start_end_ids.npy"))
        end_ids = ep_start_end_ids[:, -1]
    else:
        # Real life experiments
        # Remove camera calibration npz from iterable files
        files = [f for f in files if "camera_info.npz" not in f]

    for idx, filename in enumerate(tqdm.tqdm(files)):
        data = np.load(filename, allow_pickle=True)
        if data is None:
            continue  # Skip file

        new_size = (400, 400)
        for key in ["rgb_static", "depth_static", "rgb_gripper", "depth_gripper"]:
            img = cv2.resize(data[key], new_size)
            if "rgb" in key:
                img = img[:, :, ::-1]
                if cfg.save_viz:
                    file_dir = "./imgs/%s" % key
                    os.makedirs(file_dir, exist_ok=True)
                    img_filename = os.path.basename(filename)[:-4] + ".png"
                    img_filename = os.path.join(file_dir, img_filename)
                    cv2.imwrite(img_filename, data[key][:, :, ::-1])  # RGB
            else:
                max_depth = 4
                img = img.astype("float") / (2**16 - 1) * max_depth
                # max_range = 2**14 if "static" in key else 2**13
                img = normalizeImg(0, 1, img)
                # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            if cfg.viz:
                cv2.imshow(key, img)
                cv2.waitKey(1)


@hydra.main(config_path="../config", config_name="cfg_datacollection")
def main(cfg):
    viz_data(cfg)


if __name__ == "__main__":
    main()
