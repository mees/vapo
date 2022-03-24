import os

import cv2
import numpy as np
import pybullet as p
import torch

from vapo.affordance.utils.img_utils import resize_center, transform_and_predict, viz_aff_centers_preds
from vapo.utils.utils import init_aff_net


class TargetSearch:
    def __init__(
        self, env, mode, aff_transforms=None, aff_cfg=None, class_label=None, initial_pos=None, *args, **kwargs
    ) -> None:
        self.env = env
        self.mode = mode
        self.uniform_sample = False
        self.initial_pos = initial_pos
        self.aff_transforms = aff_transforms
        self.affordance_cfg = aff_cfg
        self.global_obs_it = 0
        self.aff_net_static_cam = init_aff_net(aff_cfg)
        self.class_label = class_label
        self.box_mask = None
        self.save_images = env.save_images

        if mode == "real_world":
            # hydra.utils.instantiate(main_cfg.cams.static_cam)
            self.static_cam = env.camera_manager.static_cam
            self.T_world_cam = self.static_cam.get_extrinsic_calibration("panda")
            self.orig_img, _ = self.static_cam.get_image()
        else:
            # self.cam_id = kwargs["cam_id"]
            self.cam_id = "static"
            self.static_cam = env.cameras[kwargs["cam_id"]]
            self.orig_img = self.env.get_obs()["rgb_obs"]["rgb_%s" % self.cam_id]

        if env.task == "pickup":
            self.box_mask, self.box_3D_end_points = self.get_box_pos_mask(env)

    def compute(self, env=None, return_all_centers=False, rand_sample=True, noisy=False):
        if env is None:
            env = self.env
        if self.mode == "real_world":
            res = self._compute_real_world(env, return_all_centers, rand_sample)
        else:
            res = self._compute_sim(env, noisy, rand_sample, return_all_centers)
        return res

    def _compute_real_world(self, env, return_all_centers, rand_sample):
        orig_img, depth_img = self.static_cam.get_image()
        self.orig_img = orig_img
        res = self._compute_target_aff(env, self.static_cam, depth_img, orig_img, rand_sample=rand_sample)
        target_pos, no_target, world_pts = res
        max_height = -1
        for pt in world_pts:
            if pt[-1] > max_height:
                target_pos = pt
                max_height = pt[-1]
        if not return_all_centers:
            res = (target_pos, no_target)
        else:
            res = (target_pos, no_target, world_pts)
        return res

    def _compute_sim(self, env, noisy, rand_sample, return_all_centers):
        obs = env.get_obs()
        depth_obs = obs["depth_obs"]["depth_%s" % self.cam_id]
        orig_img = obs["rgb_obs"]["rgb_%s" % self.cam_id]
        self.orig_img = orig_img
        if self.mode == "affordance":
            # Get environment observation
            res = self._compute_target_aff(env, self.static_cam, depth_obs, orig_img, rand_sample)
            target_pos, no_target, object_centers = res
            if noisy:
                target_pos += np.random.normal(loc=0, scale=[0.005, 0.005, 0.01], size=(len(target_pos)))
            if env.task != "pickup":
                target_pos += np.array([0.01, 0.035, -0.01])

            if return_all_centers:
                obj_centers = []
                for center in object_centers:
                    obj = {}
                    obj["target_pos"] = center
                    obj["target_str"] = self.find_env_target(env, center)
                    obj_centers.append(obj)
                res = (target_pos, no_target, obj_centers)
            else:
                res = (target_pos, no_target)
                if env.task == "pickup":
                    env.target = self.find_env_target(env, target_pos)
        else:
            if rand_sample:
                env.pick_table_obj()
            res = self._env_compute_target(env, noisy)
        return res

    def get_world_pt(self, pixel, cam, depth, env):
        x = pixel
        v, u = pixel
        if self.mode == "real_world":
            pt_cam = cam.deproject([u, v], depth, homogeneous=True)
            if pt_cam is not None and len(pt_cam) > 0:
                world_pt = self.T_world_cam @ pt_cam
                world_pt = world_pt[:3]
        else:
            if env.task == "drawer" or env.task == "slide":
                # As center might  not be exactly in handle
                # look for max depth around neighborhood
                n = 10
                depth_window = depth[x[0] - n : x[0] + n, x[1] - n : x[1] + n]
                proposal = np.argwhere(depth_window == np.min(depth_window))[0]
                v = x[0] - n + proposal[0]
                u = x[1] - n + proposal[1]
            world_pt = np.array(cam.deproject([u, v], depth))
        return world_pt

    # Env real target pos
    def _env_compute_target(self, env=None, noisy=False):
        if not env:
            env = self.env
        # This should come from static cam affordance later on
        target_pos, _ = env.get_target_pos()
        # 2 cm deviation
        target_pos = np.array(target_pos)
        if noisy:
            target_pos += np.random.normal(loc=0, scale=[0.005, 0.005, 0.01], size=(len(target_pos)))

        # always returns a target position
        no_target = False
        return target_pos, no_target

    # Aff-center model
    def _compute_target_aff(self, env, cam, depth_obs, orig_img, rand_sample=True):
        """
        orig_img (numpy.ndarray, int64): rgb, 0-255 [3 x H x W]
        """
        # Apply validation transforms
        res = transform_and_predict(
            self.aff_net_static_cam, self.aff_transforms, orig_img, class_label=self.class_label
        )
        centers, aff_mask, directions, aff_probs, object_masks = res
        # Visualize predictions
        if env.viz or self.save_images:
            img_dict = viz_aff_centers_preds(
                orig_img, aff_mask, directions, centers, "static", self.global_obs_it, viz=env.viz
            )
            if self.save_images:
                for img_path, img in img_dict.items():
                    folder = os.path.dirname(img_path)
                    os.makedirs(folder, exist_ok=True)
                    cv2.imwrite(img_path, img)
        self.global_obs_it += 1

        # No center detected
        no_target = len(centers) <= 0
        if no_target:
            default = self.initial_pos
            return np.array(default), no_target, []

        max_robustness = 0
        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class

        if rand_sample:
            target_idx = np.random.randint(len(centers))
            # target_idx = object_centers[rand_target]
        else:
            # Look for most likely center
            for i, o in enumerate(centers):
                # Mean prob of being class 1 (foreground)
                robustness = np.mean(aff_probs[object_masks == obj_class[i], 1])
                if robustness > max_robustness:
                    max_robustness = robustness
                    target_idx = i

        # World coords
        world_pts = []
        pred_shape = aff_mask.shape[:2]
        new_shape = depth_obs.shape[:2]
        for o in centers:
            o = resize_center(o, pred_shape, new_shape)
            world_pt = self.get_world_pt(o, cam, depth_obs, env)
            world_pts.append(world_pt)

        # Recover target
        if self.env.viz or self.save_images:
            v, u = resize_center(centers[target_idx], pred_shape, new_shape)
            out_img = cv2.drawMarker(
                np.array(orig_img),
                (u, v),
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=15,
                thickness=3,
                line_type=cv2.LINE_AA,
            )
            cv2.imshow("TargetSearch: img", out_img[:, :, ::-1])
            cv2.waitKey(1)
            if self.save_images:
                os.makedirs("./static_centers/", exist_ok=True)
                cv2.imwrite("./static_centers/img_%04d.jpg" % self.global_obs_it, out_img[:, :, ::-1])

        target_pos = world_pts[target_idx]
        return target_pos, no_target, world_pts

    def find_env_target(self, env, target_pos):
        min_dist = np.inf
        env_target = env.target
        for name in env.scene.table_objs:
            target_obj = env.scene.get_info()["movable_objects"][name]
            base_pos = p.getBasePositionAndOrientation(target_obj["uid"], physicsClientId=env.cid)[0]
            if p.getNumJoints(target_obj["uid"]) == 0:
                pos = base_pos
            else:  # Grasp link
                pos = p.getLinkState(target_obj["uid"], 0)[0]
            dist = np.linalg.norm(pos - target_pos)
            if dist < min_dist:
                env_target = name
                min_dist = dist
        return env_target

    def get_box_pos_mask(self, env):
        if not env:
            env = self.env
        box_top_left, box_bott_right = env.box_3D_end_points
        # Static camera
        u1, v1 = self.static_cam.project(np.array(box_top_left))
        u2, v2 = self.static_cam.project(np.array(box_bott_right))
        if self.mode == "real_world":
            shape = self.static_cam.resize_resolution
        else:
            shape = (self.static_cam.width, self.static_cam.height)
        mask = np.zeros(shape, np.uint8)
        mask = cv2.rectangle(mask, (u1, v1), (u2, v2), (1, 1, 1), thickness=-1)
        shape = (self.affordance_cfg.img_size, self.affordance_cfg.img_size)
        mask = cv2.resize(mask, shape)
        # cv2.imshow("box_mask", mask)
        # cv2.waitKey()

        # 1, H, W
        mask = torch.tensor(mask).unsqueeze(0).cuda()
        return mask, (box_top_left, box_bott_right)
