import os
import logging
import torch
import numpy as np
import cv2
import gym
from vapo.affordance.utils.img_utils import torch_to_numpy, viz_aff_centers_preds, overlay_mask
from vapo.wrappers.utils import get_obs_space, get_transforms_and_shape, \
                                    depth_preprocessing
from vapo.utils.utils import init_aff_net
from vapo.agent.core.utils import tt

logger = logging.getLogger(__name__)


class AffordanceWrapperBase(gym.Wrapper):
    def __init__(self, env, max_ts, img_size,
                 gripper_cam, static_cam, transforms=None,
                 use_pos=False,
                 affordance_cfg=None,
                 use_env_state=False,
                 real_world=False,
                 **args):
        super(AffordanceWrapperBase, self).__init__(env)
        self.env = env
        # REWARD FUNCTION
        self.affordance_cfg = affordance_cfg
        self.max_ts = max_ts
        if(self.affordance_cfg.gripper_cam.densify_reward):
            print("RewardWrapper: Gripper cam to shape reward")

        # OBSERVATION
        self.img_size = img_size

        # Prepreocessing for affordance model
        _transforms_cfg = affordance_cfg.transforms["validation"]
        _static_aff_im_size = 200
        if(img_size in affordance_cfg.static_cam):
            _static_aff_im_size = affordance_cfg.static_cam.img_size

        _gripper_aff_transforms, _aff_shape = \
            get_transforms_and_shape(_transforms_cfg, self.img_size)

        self.aff_transforms = {
            "static": get_transforms_and_shape(_transforms_cfg,
                                               self.img_size,
                                               out_size=_static_aff_im_size)[0],
            "gripper": _gripper_aff_transforms
            }

        # Preprocessing for RL policy obs
        _train_transforms, shape = get_transforms_and_shape(transforms["train"],
                                                            self.img_size)
        _val_transforms, _ = get_transforms_and_shape(transforms["validation"],
                                                      self.img_size)
        self.rl_transforms = {
            "train": _train_transforms,
            "validation": _val_transforms
        }
        self.channels = shape[0]

        # Cameras defaults
        self.obs_it = 0
        self.episode = 0
        self.save_images = env.save_images
        self.viz = env.viz

        # Parameters to define observation
        self.gripper_cam_cfg = gripper_cam
        self.static_cam_cfg = static_cam
        self.use_robot_obs = use_pos
        self.use_env_state = use_env_state
        # self._mask_transforms = DistanceTransform()

        # Parameters to store affordance
        _in_channels = _aff_shape[0]
        self.gripper_cam_aff_net = init_aff_net(affordance_cfg, 'gripper', _in_channels)
        self.static_cam_aff_net = init_aff_net(affordance_cfg, 'static', _in_channels)
        self.observation_space = get_obs_space(affordance_cfg,
                                               self.gripper_cam_cfg,
                                               self.static_cam_cfg,
                                               self.channels, self.img_size,
                                               self.use_robot_obs,
                                               self.task,
                                               real_world=real_world,
                                               oracle=self.use_env_state)

        self._curr_detected_obj = None
        self._target = None

    @property
    def target(self):
        return self.env.target

    @target.setter
    def target(self, value):
        self.env.target = value

    @property
    def curr_detected_obj(self):
        return self._curr_detected_obj

    @curr_detected_obj.setter
    def curr_detected_obj(self, world_pos):
        self._curr_detected_obj = world_pos

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        self.episode += 1
        self.obs_it = 0
        return self.observation(observation)

    def step(self, action, move_to_box=False):
        obs, reward, done, info = self.env.step(action, move_to_box)
        reward = self.reward(reward, obs, done, info["success"])
        done = self.termination(done, obs)
        # if self.curr_detected_obj is not None:
        #     self.p.addUserDebugText("ct", textPosition=self.curr_detected_obj, textColorRGB=[0, 1, 0])
        return self.observation(obs), reward, done, info

    def reward(self, rew, obs, done, success):
        if self.affordance_cfg.gripper_cam.densify_reward:
            # set by observation wrapper so that
            # both have the same observation on
            # a given timestep
            tcp_pos = obs["robot_obs"][:3]

            # If still inside area
            if(not done):
                distance = np.linalg.norm(
                        tcp_pos - self.curr_detected_obj)
                # cannot be larger than 1
                # scale dist increases as it falls away from object
                scale_dist = min(distance / self.termination_radius, 1)
                rew += (1 - scale_dist)**0.5
            else:
                # If episode was successful
                if(not success):
                    rew = self.env.reward_fail
        return rew

    def observation(self, obs):
        obs_dict = obs.copy()
        gripper_obs, gripper_viz = \
            self.get_cam_obs(obs_dict, "gripper",
                             self.gripper_cam_aff_net,
                             self.gripper_cam_cfg,
                             self.affordance_cfg.gripper_cam)
        static_obs, static_viz = \
            self.get_cam_obs(obs_dict, "static",
                             self.static_cam_aff_net,
                             self.static_cam_cfg,
                             self.affordance_cfg.static_cam)
        new_obs = {**static_obs, **gripper_obs}
        viz_dict = {**static_viz, **gripper_viz}

        # Rollout images stored by episodes(each folder is an episode)
        if(self.save_images):
            for filename, img in viz_dict.items():
                folder_name = os.path.dirname(filename)
                os.makedirs(folder_name, exist_ok=True)
                cv2.imwrite(filename, img)
        return new_obs

    def termination(self, done, obs):
        return done

    def viz_curr_target(self):
        return

    def get_images(self, obs_cfg, obs_dict, cam_type):
        raise NotImplementedError

    def get_world_pt(self, cam, pixel, depth, orig_shape):
        raise NotImplementedError

    def get_cam_obs(self, obs_dict, cam_type, aff_net,
                    obs_cfg, aff_cfg):
        obs, viz_dict = {}, {}
        depth_img, rgb_img = self.get_images(obs_cfg, obs_dict, cam_type)
        if(depth_img is not None):
            # Resize
            depth_obs = depth_preprocessing(depth_img,
                                            self.img_size)
            obs["%s_depth_obs" % cam_type] = depth_obs
        if(rgb_img is not None):
            # Img should be stored raw on replay buffer
            img_obs = np.transpose(rgb_img, (2, 0, 1))  # C, H, W
            obs["%s_img_obs" % cam_type] = img_obs

        get_gripper_target = cam_type == "gripper" and (
                    self.affordance_cfg.gripper_cam.densify_reward
                    or self.affordance_cfg.gripper_cam.target_in_obs
                    or self.affordance_cfg.gripper_cam.use_distance)

        if(aff_net is not None and (aff_cfg.use or get_gripper_target)):
            with torch.no_grad():
                # Np array 1, H, W
                processed_obs = self.aff_transforms[cam_type](tt(img_obs))
                # 1, 1, H, W in range [-1, 1]
                obs_t = processed_obs.unsqueeze(0)
                obs_t = obs_t.float().cuda()

                # 1, H, W
                _, aff_probs, aff_mask, directions = aff_net(obs_t)
                # foreground/affordance Mask
                mask = torch_to_numpy(aff_mask)
                if(get_gripper_target):
                    preds = {"%s_aff" % cam_type: aff_mask,
                             "%s_center_dir" % cam_type: directions,
                             "%s_aff_probs" % cam_type: aff_probs}

                    # Computes newest target
                    viz_dict = self.find_target_center(self.gripper_cam,
                                                       rgb_img,
                                                       depth_img,
                                                       preds)
            if(self.affordance_cfg.gripper_cam.target_in_obs):
                obs["detected_target_pos"] = self.curr_detected_obj
            if(self.affordance_cfg.gripper_cam.use_distance):
                distance = np.linalg.norm(self.curr_detected_obj
                                          - obs_dict["robot_obs"][:3]) if self.curr_detected_obj is not None else self.env.termination_radius
                obs["target_distance"] = np.array([distance])
            if(aff_cfg.use):
                obs["%s_aff" % cam_type] = mask
        return obs, viz_dict

    def transform_obs(self, obs_dct, split="validation"):
        '''
            inputs:
                obs_dct (dict): {key: torch.tensor}
        '''
        new_dct = {}
        for k, v in obs_dct.items():
            if("img_obs" in k):
                new_dct[k] = self.rl_transforms[split](v)
            else:
                new_dct[k] = v
        return new_dct

    def viz_transformed(self, obs_dct):
        ''' input:
                img: torch.tensor(shape=(C, H, W)) -1 to 1
                mask: torch.tensor(shape=(H, W, 1)) 0 or 1
        '''
        img = obs_dct["gripper_img_obs"].permute((1, 2, 0))
        img = img.detach().cpu().numpy()
        img = (img + 1) / 2
        img = (img[:, :, ::-1] * 255).astype('uint8')

        if "gripper_aff" in obs_dct:
            mask = obs_dct["gripper_aff"].squeeze()
            mask = mask.detach().cpu().numpy()
            mask = (mask * 255).astype('uint8')

            img = overlay_mask(mask, img, (0, 0, 255))
        img = cv2.resize(img, (200, 200))
        cv2.imshow("transformed img", img)

    # Aff-center
    def find_target_center(self, cam, orig_img, depth, obs):
        """
        Args:
            orig_img: np array, RGB original resolution from camera
                        shape = (1, cam.height, cam.width)
                        range = 0 to 255
                        Only used for vizualization purposes
            obs: dictionary
                - "gripper_aff":
                    affordance segmentation mask, range 0-1
                    np.array(size=(1, img_size,img_size))
                - "gripper_aff_probs":
                    affordance activation function output
                    np.array(size=(1, n_classes, img_size,img_size))
                    range 0-1
                - "gripper_center_dir": center direction predictions
                    vectors in pixel space
                    np.array(size=(1, 2, img_size,img_size))
                np array, int64
                    shape = (1, img_size, img_size)
                    range = 0 to 1
        return:
            centers: list of 3d points (x, y, z)
        """
        aff_mask = obs["gripper_aff"]
        aff_probs = obs["gripper_aff_probs"]
        directions = obs["gripper_center_dir"]
        im_dict = {}
        # Predict affordances and centers
        object_centers, center_dir, object_masks = \
            self.gripper_cam_aff_net.get_centers(aff_mask, directions)

        # Visualize predictions
        im_dict = viz_aff_centers_preds(orig_img, aff_mask,
                                        center_dir, object_centers,
                                        "gripper",
                                        self.obs_it,
                                        self.episode,
                                        viz=self.viz)
        if self.viz:
            depth_img = cv2.resize(depth, orig_img.shape[:2])
            cv2.imshow("gripper-depth", depth_img)

        if self.save_images:
            # Now between 0 and 8674
            write_depth = depth_img - depth_img.min()
            write_depth = write_depth / write_depth.max() * 255
            write_depth = np.uint8(write_depth)
            im_dict.update(
                {"./images/ep_%04d/gripper_depth/img_%04d.png"
                    % (self.episode, self.obs_it): write_depth})
            self.obs_it += 1

        # Plot different objects
        cluster_outputs = []
        object_centers = [torch_to_numpy(o) for o in object_centers]
        if(len(object_centers) <= 0):
            return im_dict

        # To numpy
        aff_probs = torch_to_numpy(aff_probs)
        aff_probs = np.transpose(aff_probs[0], (1, 2, 0))  # H, W, 2
        object_masks = torch_to_numpy(object_masks[0])  # H, W

        obj_class = np.unique(object_masks)[1:]
        obj_class = obj_class[obj_class != 0]  # remove background class

        # Look for most likely center
        n_pixels = aff_mask.shape[1] * aff_mask.shape[2]
        pred_shape = aff_probs.shape[:2]
        orig_shape = depth.shape[:2]
        for i, o in enumerate(object_centers):
            # Mean prob of being class 1 (foreground)
            cluster = aff_probs[object_masks == obj_class[i], 1]
            robustness = np.mean(cluster)
            pixel_count = cluster.shape[0] / n_pixels

            # Convert back to observation size
            o = (o * orig_shape / pred_shape).astype("int64")
            world_pt = self.get_world_pt(cam, o, depth, orig_shape)
            if(world_pt is not None):
                c_out = {"center": world_pt,
                         "pixel_count": pixel_count,
                         "robustness": robustness}
                cluster_outputs.append(c_out)

        most_robust = 0
        if(self.curr_detected_obj is not None):
            for out_dict in cluster_outputs:
                c = out_dict["center"]
                # If aff detects closer target which is large enough
                # and Detected affordance close to target
                dist = np.linalg.norm(self.curr_detected_obj - c)
                if(dist < self.env.termination_radius/2):
                    if(out_dict["robustness"] > most_robust):
                        self.curr_detected_obj = c
                        self.env.target_pos = self.curr_detected_obj
                        most_robust = out_dict["robustness"]
            if self.viz:
                self.viz_curr_target()
        return im_dict
