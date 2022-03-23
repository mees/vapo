import logging
import numpy as np
import cv2
from gym import spaces
from vapo.wrappers.affordance.aff_wrapper_base import AffordanceWrapperBase
from vapo.utils.utils import pos_orn_to_matrix, get_depth_around_point
from vapo.affordance.utils.img_utils import get_px_after_crop_resize
logger = logging.getLogger(__name__)


class AffordanceWrapperRealWorld(AffordanceWrapperBase):
    def __init__(self, *args, **kwargs):
        super(AffordanceWrapperRealWorld, self).__init__(*args, **kwargs)
        # Position and gripper action
        _action_space = np.ones(4)
        self.action_space = spaces.Box(_action_space * -1, _action_space)
        self.gripper_cam = self.env.camera_manager.gripper_cam
        self.T_tcp_cam = self.env.env.camera_manager.gripper_cam.get_extrinsic_calibration('panda')

    @property
    def task(self):
        return self.env.task

    @property
    def target_orn(self):
        return self.env.target_orn

    def step(self, action, move_to_box=False):
        if(self.task == "pickup"):
            action = np.append(action, 1)
        obs, reward, done, info = self.env.step(action, move_to_box)
        reward = self.reward(reward, obs, done, info["success"])
        return self.observation(obs), reward, done, info

    def get_images(self, obs_cfg, obs_dict, cam_type):
        depth_img, rgb_img = None, None
        if obs_cfg.use_depth:
            depth_img = obs_dict['depth_%s' % cam_type]
        if obs_cfg.use_img:
            rgb_img = obs_dict['rgb_%s' % cam_type]
        return depth_img, rgb_img

    def observation(self, obs):
        new_obs = super(AffordanceWrapperRealWorld, self).observation(obs)
        # "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        # gripper_action = int(obs["robot_obs"][-1] > 0.4)
        # new_obs.update(
        #     {"robot_obs": np.array([*obs["robot_obs"], gripper_action])})
        new_obs.update({"robot_obs": obs["robot_obs"]})
        return new_obs

    def get_world_pt(self, cam, pixel, depth, orig_shape):
        o, depth_non_zero = get_depth_around_point(pixel, depth)
        if depth_non_zero:
            cam_frame_pt = cam.deproject(o, depth)
            tcp_pos, tcp_orn = self.env.robot.get_tcp_pos_orn()
            tcp_mat = pos_orn_to_matrix(tcp_pos, tcp_orn)
            world_pt = tcp_mat @ self.T_tcp_cam @ np.array([*cam_frame_pt, 1])
            world_pt = world_pt[:3]
        else:
            world_pt = None
        return world_pt

    def viz_curr_target(self):
        u, v = self.target_search.static_cam.project(self.curr_detected_obj)
        u, v = get_px_after_crop_resize((u, v),
                                        self.target_search.static_cam.crop_coords,
                                        self.target_search.static_cam.resize_resolution)
        img = self.target_search.orig_img.copy()
        img = cv2.drawMarker(img, (int(u), int(v)),
                             (0, 0, 0),
                             markerType=cv2.MARKER_CROSS,
                             markerSize=15,
                             thickness=3,
                             line_type=cv2.LINE_AA)
        cv2.imshow("detected target", img[:, :, ::-1])
