import logging

import cv2
import hydra
import numpy as np
import pybullet as p

from vapo.affordance.dataset_creation.core.utils import create_data_ep_split, create_json_file
from vapo.affordance.dataset_creation.data_labeler import DataLabeler
from vapo.affordance.utils.img_utils import add_img_text, resize_center

log = logging.getLogger(__name__)


class DataLabelerLang(DataLabeler):
    def __init__(self, cfg, classifier=None, discovery_episodes=[], *args, **kwargs):
        super(DataLabelerLang, self).__init__(cfg, *args, **kwargs)
        self.saved_static_frames = set()
        self._last_frame_task = None
        self._last_frame_lang_ann = None
        self.curr_task = {"static": None, "gripper": None}
        self._project_pt = None
        self.env = hydra.utils.instantiate(self.teleop_cfg.env)

    def get_contact_info(self, data):
        if self.real_world:
            return True
        else:
            obs = self.env.reset(robot_obs=data["robot_obs"], scene_obs=data["scene_obs"])
            static_reset = obs["rgb_obs"]["rgb_static"]
            static_file = data["rgb_static"]
            img = np.hstack([static_reset, static_file])
            contact_pts = np.array(p.getContactPoints())[:, 1]
            contact = (contact_pts == self.env.robot.robot_uid).any()
            # # Print text
            if self.viz:
                text_label = "contact" if contact else "no contact"
                img = add_img_text(img, text_label)
                cv2.imshow("reset/file img", img[:, :, ::-1])
                cv2.waitKey(1)
            return contact

    def open_to_closed(self, dct):
        """
        dct =  {"robot_obs": robot_obs,
                "last_pt": last_pt,
                "frame_idx": frame_idx,
                "data": data}
        """
        curr_robot_obs = dct["robot_obs"]
        last_pt = dct["last_pt"]
        frame_idx = dct["frame_idx"]
        contact = self.get_contact_info(dct["data"])

        curr_pt = curr_robot_obs[:3]
        if contact:
            self.label_gripper(self.img_hist["gripper"], curr_pt, last_pt, contact)
            self.label_static(self.img_hist["static"], curr_robot_obs)
            self.img_hist = {"static": [], "gripper": []}

    def closed_gripper(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_pt": last_pt,
                "data": data} # np file of current robot obs
        """
        curr_robot_obs = dct["robot_obs"]
        last_pt = dct["last_pt"]

        curr_pt = curr_robot_obs[:3]
        contact = self.get_contact_info(dct["data"])
        if contact:
            if len(self.img_hist["static"]) > 1:
                self.label_static(self.img_hist["static"], curr_robot_obs)
                self.img_hist["static"] = []
            self.label_gripper(self.img_hist["gripper"], curr_pt, last_pt, contact)
            self.img_hist = {"static": [], "gripper": []}

    def closed_to_open(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_pt": last_pt,
                "frame_idx": frame_idx}
        """
        # self._project_pt = dct["last_pt"]
        # self.label_static(self.img_hist["static"], dct["robot_obs"])
        return

    def after_loop(self, episode=0):
        self.save_data(episode)
        min_labels = 1
        if self.single_split:
            create_json_file(self.output_dir, self.save_split, min_labels, only_language=True)
        else:
            create_data_ep_split(
                self.output_dir,
                self.labeling.split_by_episodes,
                min_labels,
                task_discovery_ep=self.task_discovery_folders,
                only_language=True,
            )

    def after_iteration(self, episode, ep_id, curr_folder):
        # log.info("Saving information... current episode %d " % episode)
        if np.sum([len(v) for v in self.save_dict.values()]) > self.frames_before_saving:
            self.save_data(episode)

        # log.info("Saved frames")
        return False

    def label_gripper(self, img_hist, curr_pt, last_pt, contact):
        cam = "gripper"
        out_img_size = self.output_size[cam]
        save_dict = {}
        self._last_task_centers = None
        for idx, (fr_idx, ep_id, im_id, robot_obs, img) in enumerate(img_hist):
            # Shape: [H x W x 2]
            H, W = out_img_size  # img.shape[:2]
            centers = []

            # Get lang ann
            if ep_id in self.lang_ann:
                # lang_ann has sets
                task = list(self.lang_ann[ep_id].keys())[-1]
                lang_ann = list(self.lang_ann[ep_id][task])
                if self.curr_task[cam] is None or self.curr_task[cam] not in task:
                    self.curr_task[cam] = task
            else:
                lang_ann = []
                task = []
                self.curr_task[cam] = None

            # Gripper width
            if robot_obs[-1] > self.gripper_width_tresh or contact:
                if curr_pt is not None:
                    # Center in matrix convention (row, column)
                    mask, center_px = self.get_gripper_mask(img, robot_obs[:-1], curr_pt)
                    center_px = resize_center(center_px, mask.shape[:2], out_img_size)
                    if np.all(center_px > 0) and np.all(center_px < H):
                        # Only one class, label as one
                        centers.append([0, *center_px])
            else:
                centers = []

            # Visualize results
            img = cv2.resize(img, out_img_size)
            caption = "" if len(lang_ann) == 0 else lang_ann[-1]
            out_img = self.viz_imgs(img, centers, caption, cam_str=cam)
            centers = np.zeros((0, 3)) if len(centers) < 1 else np.stack(centers)

            save_dict[im_id] = {
                "frame": img,
                "centers": centers,
                "lang_ann": lang_ann,
                "task": task,
                "viz_out": out_img,
                "gripper_width": robot_obs[-1],
            }
        self.save_dict[cam].update(save_dict)

    def label_static(self, static_hist, curr_robot_obs):
        cam = "static"
        back_min, back_max = self.back_frames
        out_img_size = self.output_size[cam]
        save_dict = {}
        H, W = self.output_size[cam]
        pt = curr_robot_obs[:3]
        self._project_pt = pt

        _ep_id = self.img_hist["static"][-1][1]
        # Get lang ann
        if _ep_id in self.lang_ann:
            # lang_ann has sets
            task = list(self.lang_ann[_ep_id].keys())[-1]
            lang_ann = list(self.lang_ann[_ep_id][task])
        else:
            lang_ann = []
            task = ""
            self.curr_task[cam] = None

        for idx, (fr_idx, ep_id, im_id, robot_obs, img) in enumerate(static_hist):
            if im_id in self.saved_static_frames:
                continue
            # For static mask assume oclusion
            # until back_frames_min before
            centers = []

            # first create fp masks and place current(newest)
            # mask and optical flow on top
            label = self.classifier.predict(curr_robot_obs)

            # Get centers
            if idx <= len(static_hist) - back_min and idx > len(static_hist) - back_max:
                # Get new grip
                mask, center_px = self.get_static_mask(img, self._project_pt)
                center_px = resize_center(center_px, mask.shape[:2], out_img_size)
                if np.all(center_px > 0) and np.all(center_px < H):
                    # Only one class, label as one
                    centers.append([label, *center_px])

            img = cv2.resize(img, out_img_size)

            if len(centers) > 0:
                centers = np.stack(centers)
            else:
                centers = np.zeros((0, 2))

            caption = "" if len(lang_ann) == 0 else lang_ann[-1]
            out_img = self.viz_imgs(img, centers, caption, cam_str=cam)

            save_dict[im_id] = {
                "frame": img,
                "centers": centers,
                "lang_ann": lang_ann,
                "task": task,
                "viz_out": out_img,
            }
            self.saved_static_frames.add(im_id)
        self.save_dict[cam].update(save_dict)

    def viz_imgs(self, rgb_img, centers, caption="", cam_str=""):
        """
        :param rgb_img(np.ndarray): H, W, C
        :param centers(np.ndarray): shape=(n_centers, 3). label, u, v = centers[i]
        """
        # Visualize results
        out_img = rgb_img.copy()

        for c in centers:
            label, v, u = c
            color = self.classifier.colors[label][:3]
            color = [int(c_item * 255) for c_item in color]
            out_img = cv2.drawMarker(
                np.array(out_img),
                (u, v),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        if self.viz:
            viz_size = (300, 300)
            out_img = cv2.resize(out_img, viz_size)
            out_img = add_img_text(out_img, caption)
            cv2.imshow("%s" % cam_str, out_img[:, :, ::-1])
            cv2.waitKey(1)
        return out_img


@hydra.main(config_path="../../config", config_name="cfg_datacollection")
def main(cfg):
    labeler = DataLabelerLang(cfg)
    labeler.iterate()


if __name__ == "__main__":
    main()
