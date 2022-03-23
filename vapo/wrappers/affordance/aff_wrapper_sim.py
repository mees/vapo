import numpy as np
import cv2
import logging
import gym.spaces as spaces
from vapo.wrappers.affordance.aff_wrapper_base import AffordanceWrapperBase
logger = logging.getLogger(__name__)


class AffordanceWrapperSim(AffordanceWrapperBase):
    def __init__(self, *args, **kwargs):
        super(AffordanceWrapperSim, self).__init__(*args, **kwargs)
        self.env.termination_radius = kwargs['max_target_dist']
        self.termination_radius = kwargs['max_target_dist']
        self.use_aff_termination = kwargs['use_aff_termination']
        self.task = self.env.task

        # Observation
        self.cam_ids = self.env.cam_ids

        # Action Space
        if(self.env.task == "pickup"):
            # 0-2 -> position , 3 -> yaw angle 4 gripper action
            _action_space = np.ones(5)
        else:
            # 0-2 -> position , 3-5 -> orientation, 6 gripper action
            _action_space = np.ones(7)
        self.action_space = spaces.Box(_action_space * -1, _action_space)
        self.gripper_cam = self.cameras[self.cam_ids["gripper"]]

        # Debug
        self.target_search = None

    def viz_curr_target(self):
        ''' See current target on static camera'''
        u, v = self.target_search.static_cam.project(self.curr_detected_obj)
        img = self.target_search.orig_img.copy()
        img = cv2.drawMarker(img, (int(u), int(v)),
                             (255, 0, 0),
                             markerType=cv2.MARKER_CROSS,
                             markerSize=15,
                             thickness=3,
                             line_type=cv2.LINE_AA)
        cv2.imshow("detected target", img[:, :, ::-1])

    def get_images(self, obs_cfg, obs_dict, cam_type):
        depth_img, rgb_img = None, None
        if obs_cfg.use_depth:
            depth_img = obs_dict['depth_obs']["depth_%s" % cam_type]
        if obs_cfg.use_img:
            rgb_img = obs_dict['rgb_obs']["rgb_%s" % cam_type]
        return depth_img, rgb_img

    def get_world_pt(self, cam, pixel, depth, orig_shape):
        if(self.env.task == "drawer" or self.env.task == "slide"):
            # As center might  not be exactly in handle
            # look for max depth around neighborhood
            n = 10
            uv_max = np.clip(pixel + [n, n], 0, orig_shape)
            uv_min = np.clip(pixel - [n, n], 0, orig_shape)
            depth_window = depth[uv_min[0]:uv_max[0],
                                 uv_min[1]:uv_max[1]]
            proposal = np.argwhere(depth_window == np.min(depth_window))[0]
            v = pixel[0] - n + proposal[0]
            u = pixel[1] - n + proposal[1]
        else:
            v, u = pixel
        world_pt = cam.deproject([u, v], depth)
        return world_pt

    def observation(self, obs):
        # Store global images (all cameras)
        new_obs = super(AffordanceWrapperSim, self).observation(obs)
        self.env.save_and_viz_obs(obs)
        # "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        if(self.use_robot_obs):
            if(self.task == "pickup"):
                # *tcp_pos(3), *tcp_euler(1) z angle ,
                # gripper_opening_width(1), gripper_action
                new_obs["robot_obs"] = np.array([*obs["robot_obs"][:3],
                                                 *obs["robot_obs"][5:7],
                                                 obs["robot_obs"][-1]])
            else:
                # *tcp_pos(3), *tcp_euler(3),
                # gripper_opening_width(1), gripper_action
                new_obs["robot_obs"] = np.array([*obs["robot_obs"][:7],
                                                 obs["robot_obs"][-1]])
        return new_obs

    def termination(self, done, obs):
        if(self.use_aff_termination):
            distance = np.linalg.norm(self.curr_detected_obj
                                      - obs["robot_obs"][:3])
        else:
            # Real distance
            target_pos, _ = self.env.get_target_pos()
            distance = np.linalg.norm(target_pos
                                      - obs["robot_obs"][:3])
        return done or distance > self.termination_radius
