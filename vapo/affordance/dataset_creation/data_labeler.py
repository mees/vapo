import json
import logging
import os

import vapo.affordance.utils.flowlib as flowlib
import cv2
import hydra
import numpy as np
import pybullet as p

from vapo.affordance.dataset_creation.core.data_classifier import BaseDetector
from vapo.affordance.dataset_creation.core.data_reader import DataReader
from vapo.affordance.dataset_creation.core.utils import (
    create_data_ep_split,
    create_json_file,
    instantiate_env,
    save_dict_data,
)
from vapo.affordance.utils.img_utils import (
    create_circle_mask,
    get_px_after_crop_resize,
    overlay_flow,
    overlay_mask,
    resize_mask_and_center,
    tresh_np,
)
from vapo.affordance.utils.utils import get_abs_path

log = logging.getLogger(__name__)


class DataLabeler(DataReader):
    def __init__(self, cfg, classifier=None, discovery_episodes=[], new_cfg=False, *args, **kwargs):
        super(DataLabeler, self).__init__(cfg, *args, **kwargs)
        self.remove_blank_mask_instances = cfg.labeling.remove_blank_mask_instances
        self.save_viz = cfg.save_viz
        self.viz = cfg.viz
        self.mask_on_close = cfg.mask_on_close
        self.fixed_pt_del_radius = cfg.labeling.fixed_pt_del_radius
        self.real_world = cfg.labeling.real_world
        self.back_frames = cfg.labeling.back_frames  # [min, max]
        self.label_size = cfg.labeling.label_size
        self.output_size = {k: (cfg.output_size[k], cfg.output_size[k]) for k in ["gripper", "static"]}
        self.pixel_indices = {
            "gripper": np.indices(self.output_size["gripper"], dtype=np.float32).transpose(1, 2, 0),
            "static": np.indices(self.output_size["static"], dtype=np.float32).transpose(1, 2, 0),
        }
        _static, _gripper, _teleop_cfg = instantiate_env(cfg, self.real_world, new_cfg=new_cfg)
        self.teleop_cfg = _teleop_cfg
        self.static_cam = _static
        self.gripper_cam = _gripper
        self.save_dict = {"gripper": {}, "static": {}}
        self.output_dir = get_abs_path(cfg.output_dir)
        self._fixed_points = []
        self.labeling = cfg.labeling
        self.single_split = cfg.output_cfg.single_split
        self.save_split = cfg.output_cfg.save_split
        self.frames_before_saving = cfg.frames_before_saving
        self.classifier = classifier if classifier is not None else BaseDetector(cfg.task_detector)

        self.gripper_width_tresh = 0.015
        self.task_discovery_folders = discovery_episodes
        log.info("Writing to %s" % self.output_dir)

    @property
    def fixed_points(self):
        return self._fixed_points

    @fixed_points.setter
    def fixed_points(self, value):
        self._fixed_points = value

    def after_loop(self, episode=-1):
        self.save_data(episode)
        if self.single_split:
            create_json_file(
                self.output_dir,
                self.save_split,
                self.labeling.min_labels,
                remove_blank_mask_instances=self.remove_blank_mask_instances,
                n_classes=self.classifier.n_classes,
            )
        else:
            create_data_ep_split(
                self.output_dir,
                self.labeling.split_by_episodes,
                self.labeling.min_labels,
                task_discovery_ep=self.task_discovery_folders,
                remove_blank_mask_instances=self.remove_blank_mask_instances,
                n_classes=self.classifier.n_classes,
            )
        # Add n_classes and orientation per class
        # data = {"target_orn": None}
        orientations = {}
        for label, v in self.classifier.clusters.items():
            orientations[int(label)] = list(v["orn"])
        new_data = {"target_orn": orientations}
        output_path = os.path.join(self.output_dir, "episodes_split.json")
        with open(output_path, "r+") as outfile:
            data = json.load(outfile)
            data.update(new_data)
            outfile.seek(0)
            json.dump(data, outfile, indent=2)

    def after_iteration(self, episode, ep_id, curr_folder):
        if np.sum([len(v) for v in self.save_dict.values()]) > self.frames_before_saving:
            self.save_data(episode)
        # dont stop
        return False

    def on_episode_end(self, episode):
        self.save_data(episode)

    def closed_gripper(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_pt": last_pt,
                "data": data}
        """
        curr_robot_obs = dct["robot_obs"]
        last_pt = dct["last_pt"]
        if self.mask_on_close:
            curr_pt = curr_robot_obs[:3]
            self.label_gripper(self.img_hist["gripper"], curr_pt, last_pt)
        super().closed_gripper(dct)

    def closed_to_open(self, dct):
        """
        dct = {"robot_obs": robot_obs,
                "last_pt": last_pt,
                "frame_idx": frame_idx}
        """
        new_pt = (dct["frame_idx"], dct["robot_obs"])
        self.fixed_points.append(new_pt)

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

        curr_pt = curr_robot_obs[:3]
        self.label_gripper(self.img_hist["gripper"], curr_pt, last_pt)
        self.label_static(self.img_hist["static"], curr_robot_obs)
        self.fixed_points = self.update_fixed_points(curr_robot_obs, frame_idx)
        super().open_to_closed(dct)

    def save_data(self, episode):
        for cam_str in ["static", "gripper"]:
            save_dir = os.path.join(self.output_dir, "episode_%02d" % episode)
            save_dict_data(self.save_dict[cam_str], save_dir, sub_dir="%s_cam" % cam_str, save_viz=self.save_viz)
        self.save_dict = {"gripper": {}, "static": {}}

    def label_gripper(self, img_hist, curr_pt, last_pt):
        out_img_size = self.output_size["gripper"]
        save_dict = {}
        for idx, (fr_idx, ep_id, im_id, robot_obs, img) in enumerate(img_hist):
            # Shape: [H x W x 2]
            H, W = out_img_size  # img.shape[:2]
            directions = np.stack([np.ones((H, W)), np.zeros((H, W))], axis=-1).astype(np.float32)
            mask = np.zeros(out_img_size)
            centers = []
            # Gripper width
            if robot_obs[-1] > self.gripper_width_tresh:
                for point in [curr_pt, last_pt]:
                    if point is not None:
                        # Center and directions in matrix convention (row, column)
                        new_mask, center_px = self.get_gripper_mask(img, robot_obs[:-1], point)
                        new_mask, center_px = resize_mask_and_center(new_mask, center_px, out_img_size)
                        if np.any(center_px < 0) or np.any(center_px >= H):
                            new_mask = np.zeros(out_img_size)
                        else:
                            # Only one class, label as one
                            centers.append([0, *center_px])
                        directions = self.label_directions(center_px, new_mask, directions, "gripper")
                        mask = overlay_mask(new_mask, mask, (255, 255, 255))
            else:
                mask = np.zeros(out_img_size)
                centers = []

            # Visualize results
            img = cv2.resize(img, out_img_size)
            mask = np.expand_dims(mask, 0)
            out_img, flow_over_img = self.viz_imgs(img, mask, directions, centers, cam_str="gripper")
            centers = np.zeros((0, 3)) if len(centers) < 1 else np.stack(centers)
            save_dict[im_id] = {
                "frame": img,
                "mask": mask,
                "centers": centers,
                "directions": directions,
                "viz_out": out_img,
                "viz_dir": flow_over_img,
                "gripper_width": robot_obs[-1],
            }
        self.save_dict["gripper"].update(save_dict)

    def viz_imgs(self, rgb_img, aff_mask, directions, centers, separate_masks=None, cam_str=""):
        """
        :param rgb_img(np.ndarray): H, W, C
        :param aff_mask(np.ndarray): n_classes, H, W, Each channel has a binary (0-255) mask for the given class. Class 0 is background
        :param directions(np.ndarray): H, W, 2 labeled directions for foreground mask
        :param centers(np.ndarray): shape=(n_centers, 3). label, u, v = centers[i]
        """
        # Visualize results
        out_img = rgb_img
        for label in range(aff_mask.shape[0]):
            color = self.classifier.colors[label]
            color = tuple((color * 255).astype("int32"))
            out_img = overlay_mask(aff_mask[label], out_img, color)

        flow_img = flowlib.flow_to_image(directions)

        fg_mask = aff_mask.any(axis=0).astype("uint8") * 255  # Binary
        flow_over_img = overlay_flow(flow_img, rgb_img, fg_mask)

        for c in centers:
            label, v, u = c
            color = self.classifier.colors[label][:3]
            color = [int(c_item * 255) for c_item in color]
            flow_over_img = cv2.drawMarker(
                np.array(flow_over_img),
                (u, v),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        if self.viz:
            viz_size = (200, 200)
            viz_img = cv2.resize(out_img, viz_size)

            viz_flow_over_img = cv2.resize(flow_over_img, viz_size)
            cv2.imshow("%s" % cam_str, viz_img[:, :, ::-1])
            cv2.imshow("%s flow_img" % cam_str, viz_flow_over_img[:, :, ::-1])
            # cv2.imshow('Gripper real flow', viz_flow)
            if separate_masks is not None:
                fp_mask, np_mask = separate_masks
                out_separate = overlay_mask(fp_mask, rgb_img, (255, 0, 0))
                out_separate = overlay_mask(np_mask, out_separate, (0, 0, 255))
                out_separate = cv2.resize(out_separate, viz_size)
                cv2.imshow("Separate", out_separate[:, :, ::-1])
            cv2.waitKey(1)
        return out_img, flow_over_img

    def label_static(self, static_hist, curr_robot_obs):
        cam = "static"
        back_min, back_max = self.back_frames
        out_img_size = self.output_size[cam]
        save_dict = {}
        for idx, (fr_idx, ep_id, im_id, _, img) in enumerate(static_hist):
            # For static mask assume oclusion
            # until back_frames_min before
            centers = []
            H, W = self.output_size[cam]  # img.shape[:2]  # img_shape = (H, W, C)
            directions = np.stack([np.ones((H, W)), np.zeros((H, W))], axis=-1).astype(np.float32)
            full_mask, centers_px, fp_directions = self.update_mask(
                np.zeros((self.classifier.n_classes, H, W), dtype=np.uint8), directions, (fr_idx, img)
            )

            # first create fp masks and place current(newest)
            # mask and optical flow on top
            label = self.classifier.predict(curr_robot_obs)
            if idx <= len(static_hist) - back_min and idx > len(static_hist) - back_max:
                # Get new grip
                pt = curr_robot_obs[:3]
                mask, center_px = self.get_static_mask(img, pt)
                mask, center_px = resize_mask_and_center(mask, center_px, out_img_size)
                directions = self.label_directions(center_px, mask, fp_directions, "static")
                centers.append([label, *center_px])
            else:
                # No segmentation in current image due to occlusion
                mask = np.zeros((H, W))

            # Join new mask to fixed points
            # label_mask = np.stack((full_mask[label], mask)).any(axis=0).astype("uint8")
            # full_mask[label] = label_mask * 255.0

            img = cv2.resize(img, out_img_size)
            full_mask[label] = overlay_mask(mask, full_mask[label], (255, 255, 255))

            centers += centers_px  # Concat to list
            if len(centers) > 0:
                centers = np.stack(centers)
            else:
                centers = np.zeros((0, 2))
            out_img, flow_over_img = self.viz_imgs(
                img,
                full_mask,
                directions,
                centers,
                # separate_masks=(fp_mask, mask),
                cam_str="static",
            )

            save_dict[im_id] = {
                "frame": img,
                "mask": full_mask,  # 0-255
                "centers": centers,
                "directions": directions,
                "viz_out": out_img,
                "viz_dir": flow_over_img,
            }
        self.save_dict["static"].update(save_dict)

    def label_directions(self, center, object_mask, direction_labels, camtype):
        # Shape: [H x W x 2]
        indices = self.pixel_indices[camtype]
        object_mask = tresh_np(object_mask, 100)
        object_center_directions = (center - indices).astype(np.float32)
        object_center_directions = object_center_directions / np.maximum(
            np.linalg.norm(object_center_directions, axis=2, keepdims=True), 1e-10
        )

        # Add it to the labels
        direction_labels[object_mask == 1] = object_center_directions[object_mask == 1]
        return direction_labels

    def update_mask(self, mask, directions, frame_img_tuple):
        """
        :param mask(np.ndarray): N_classes, H, W
        """
        # Update masks with fixed_points
        centers = []
        (frame_timestep, img) = frame_img_tuple
        for point_timestep, pt in self.fixed_points:
            # Only add point if it was fixed before seing img
            if frame_timestep >= point_timestep:
                label = self.classifier.predict(pt)

                new_mask, center_px = self.get_static_mask(img, pt[:3])
                new_mask, center_px = resize_mask_and_center(new_mask, center_px, self.output_size["static"])
                centers.append([label, *center_px])
                directions = self.label_directions(center_px, new_mask, directions, "static")
                mask[label] = overlay_mask(new_mask, mask[label], (255, 255, 255))
        return mask, centers, directions

    def update_fixed_points(self, new_point, current_frame_idx):
        x = []
        radius = self.fixed_pt_del_radius
        for frame_idx, pt in self.fixed_points:
            if np.linalg.norm(new_point[:3] - pt[:3]) > radius:
                x.append((frame_idx, pt))
        # x = [ p for (frame_idx, p) in fixed_points if
        # ( np.linalg.norm(new_point - p) > radius)]
        # # and current_frame_idx - frame_idx < 100 )
        return x

    def get_static_mask(self, static_im, point):
        # Img history containes previus frames where gripper action was open
        # Point is the point in which the gripper closed for the 1st time
        # TCP in homogeneus coord.
        point = np.append(point, 1)

        # Project point to camera
        # x,y <- pixel coords
        tcp_x, tcp_y = self.static_cam.project(point)
        if self.real_world:
            tcp_x, tcp_y = get_px_after_crop_resize(
                (tcp_x, tcp_y),
                self.static_cam.crop_coords,
                self.static_cam.resize_resolution,
            )
        static_mask = create_circle_mask(static_im, (tcp_x, tcp_y), r=self.label_size["static"])
        return static_mask, (tcp_y, tcp_x)  # matrix coord

    def get_gripper_mask(self, img, robot_obs, point):
        pt, orn = robot_obs[:3], robot_obs[3:]
        if self.real_world:
            orn = p.getQuaternionFromEuler(orn)
            transform_matrix = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
            transform_matrix = np.vstack([transform_matrix, np.zeros(3)])
            tcp2global = np.hstack([transform_matrix, np.expand_dims(np.array([*pt, 1]), 0).T])
            global2tcp = np.linalg.inv(tcp2global)
            point = global2tcp @ np.array([*point, 1])
            tcp_pt = point[:3]

            # Transform pt to homogeneus cords and project
            tcp_x, tcp_y = self.gripper_cam.project(tcp_pt)

            # Get img coords after resize
            tcp_x, tcp_y = get_px_after_crop_resize(
                (tcp_x, tcp_y),
                self.gripper_cam.crop_coords,
                self.gripper_cam.resize_resolution,
            )

        else:
            orn = p.getQuaternionFromEuler(orn)
            tcp2cam_pos, tcp2cam_orn = self.gripper_cam.tcp2cam_T
            # cam2tcp_pos = [0.1, 0, -0.1]
            # cam2tcp_orn = [0.430235, 0.4256151, 0.559869, 0.5659467]
            cam_pos, cam_orn = p.multiplyTransforms(pt, orn, tcp2cam_pos, tcp2cam_orn)

            # Create projection and view matrix
            cam_rot = p.getMatrixFromQuaternion(cam_orn)
            cam_rot = np.array(cam_rot).reshape(3, 3)
            cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]

            # Extrinsics change as robot moves
            self.gripper_cam.viewMatrix = p.computeViewMatrix(cam_pos, cam_pos + cam_rot_y, -cam_rot_z)

            # Transform pt to homogeneus cords and project
            point = np.append(point, 1)
            tcp_x, tcp_y = self.gripper_cam.project(point)

        if tcp_x > 0 and tcp_y > 0:
            mask = create_circle_mask(img, (tcp_x, tcp_y), r=self.label_size["gripper"])
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 1))
        return mask, (tcp_y, tcp_x)


@hydra.main(config_path="../../config", config_name="cfg_datacollection")
def main(cfg):
    labeler = DataLabeler(cfg)
    labeler.iterate()
    # labeler.after_loop()


if __name__ == "__main__":
    main()
