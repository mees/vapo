import glob
import json
import os

import cv2
import tqdm

from vapo.utils.utils import get_files


def make_video(files, fps=60, video_name="v"):
    h, w, c = cv2.imread(files[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, fps, (w, h))  # 30 fps
    print("writing video to %s" % video_name)
    for f in tqdm.tqdm(files):
        img = cv2.imread(f)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()


def join_val_ep(dataset_dir, cam):
    json_file = os.path.join(dataset_dir, "episodes_split.json")
    with open(json_file) as f:
        ep_data = json.load(f)
    files = []
    for ep in ep_data["validation"].keys():
        img_folder = os.path.join(dataset_dir, ep + "/viz_out/%s_cam/" % cam)
        files += get_files(img_folder, "png")
    files.sort()
    return files


def make_videos(path, cam, val_dir=False):
    if val_dir:
        video_name = os.path.join(os.path.dirname(path), os.path.basename(path) + "_validation.mp4")
        files = join_val_ep(path, cam)
        make_video(files, fps=60, video_name=video_name)
    else:
        for img_folder in path:
            files = get_files(img_folder, "png")[:-1]
            if not files:
                return
            video_name = os.path.join(os.path.dirname(img_folder), os.path.basename(img_folder) + ".mp4")
            make_video(files, fps=15, video_name=video_name)


if __name__ == "__main__":
    pred_folder = "/mnt/ssd_shared/Users/Jessica/Documents/rollouts/2022-03-07/11-20-59/images"

    gripper_ep_dirs = glob.glob("%s/ep_*" % pred_folder)
    gripper_eps = [os.path.join(ep_dir, "gripper_dirs") for ep_dir in gripper_ep_dirs]
    dirs = ["%s/render_orig" % pred_folder]
    dirs.extend(gripper_eps)
    # dirs.extend("%s/gripper_orig" % pred_folder)
    make_videos(dirs, "")
