import os
import logging
import math
import numpy as np
import gym
import torch
import pybullet as p
from vapo.wrappers.utils import find_cam_ids

import cv2

from gym import spaces
from vr_env.envs.play_table_env import PlayTableSimEnv
from vr_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
from vapo.wrappers.play_table_rand_scene import PlayTableRandScene
from vapo.utils.utils import get_3D_end_points, pos_orn_to_matrix
logger = logging.getLogger(__name__)


class PlayTableRL(PlayTableSimEnv):
    def __init__(self, task="slide", sparse_reward=False,
                 max_counts=50, viz=False, save_images=False, **args):
        if('use_egl' in args and args['use_egl']):
            # if("CUDA_VISIBLE_DEVICES" in os.environ):
            #     device_id = os.environ["CUDA_VISIBLE_DEVICES"]
            #     device = int(device_id)
            # else:
            device = torch.cuda.current_device()
            device = torch.device(device)
            self.set_egl_device(device)
        super(PlayTableRL, self).__init__(**args)
        self.task = task
        _action_space = np.ones(7)
        self.action_space = spaces.Box(_action_space * -1, _action_space)
        obs_space_dict = {
            "scene_obs": gym.spaces.Box(low=0, high=1.5, shape=(3,)),
            'robot_obs': gym.spaces.Box(low=-0.5, high=0.5, shape=(7,)),
            'rgb_obs': gym.spaces.Box(low=0, high=255, shape=(3, 300, 300)),
            'depth_obs': gym.spaces.Box(low=0, high=255, shape=(1, 300, 300))
        }
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        self.sparse_reward = sparse_reward
        self.offset = np.array([*args["offset"], 1])
        self.reward_fail = args['reward_fail']
        self.reward_success = args['reward_success']
        self._obs_it = 0
        self.viz = viz
        self.save_images = save_images
        self.cam_ids = find_cam_ids(self.cameras)

        self._rand_scene = "rand_scene" in args
        _initial_obs = self.get_obs()["robot_obs"]
        self._start_orn = _initial_obs[3:6]

        if(task == "pickup"):
            if(self._rand_scene):
                load_only_one = args["rand_scene"]["load_only_one"]
            else:
                load_only_one = False
            self.scene = PlayTableRandScene(p=self.p, cid=self.cid,
                                            np_random=self.np_random,
                                            load_only_one=load_only_one,
                                            max_counts=max_counts,
                                            **args['scene_cfg'])
            self.rand_positions = self.scene.rand_positions
            self.load()

            self._target = self.scene.target
            # x1,y1,z1, width, height, depth (x,y,z) in meters]
            self.box_pos = self.scene.object_cfg['fixed_objects']['bin']["initial_pos"]
            w, h, d = 0.24, 0.4, 0.08
            self.box_3D_end_points = get_3D_end_points(*self.box_pos, w, h, d)
        else:
            self._target = task
            self.rand_positions = False

    @property
    def obs_it(self):
        return self._obs_it

    @property
    def rand_scene(self):
        return self._rand_scene

    @property
    def start_orn(self):
        return self._start_orn

    @property
    def target(self):
        return self._target

    @obs_it.setter
    def obs_it(self, value):
        self._obs_it = value

    @target.setter
    def target(self, value):
        self._target = value
        self.scene.target = value

    def pick_table_obj(self, eval=False):
        if self.task == "pickup":
            self.scene.pick_table_obj(eval)
            self.target = self.scene.target

    def get_scene_with_objects(self, obj_lst, load_scene=False):
        self.scene.get_scene_with_objects(obj_lst, load_scene)
        self.target = self.scene.target

    def pick_rand_scene(self, objs_success=None, load=False, eval=False):
        self.scene.pick_rand_scene(objs_success, load, eval)

    def reset(self, eval=False):
        if(self.task == "pickup"):
            if(self.rand_scene and not eval):
                self.pick_rand_scene()
        # Resets scene, robot, etc
        res = super(PlayTableRL, self).reset()
        self.pick_table_obj(eval)
        return res

    def set_egl_device(self, device):
        assert "EGL_VISIBLE_DEVICES" not in os.environ, "Do not manually set EGL_VISIBLE_DEVICES"
        cuda_id = device.index if device.type == "cuda" else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to VREnv README"
            )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def step(self, action, *args):
        # Action space that SAC sees is between -1,1 for all elements in vector
        update_target = len(action) == 3
        if(len(action) == 5):
            a = action.copy()
            if(self.task == "pickup"):  # Constraint angle
                a = [*a[:3], 0, 0, a[-2], a[-1]]
                # Scale vector to true values that it can take
                a = self.robot.relative_to_absolute(a)
                a = list(a)
                # constraint angle
                a[1] = np.array([- math.pi, 0, a[1][-1]])
                update_target = False
        else:
            a = action
        self.robot.apply_action(a, update_target)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        # dict w/keys: "rgb_obs", "depth_obs", "robot_obs","scene_obs"
        done = self._termination()
        if(done and self.task == "pickup"):
            success = self.check_success()
        else:
            success = done
        obs = self.get_obs()
        reward, r_info = self._reward(success)
        info = self.get_info()
        info.update({"success": success,
                     **r_info})
        # obs, reward, done, info
        return obs, reward, done, info

    def _normalize(self, val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    def _reward(self, success):
        # returns the normalized state
        targetWorldPos, targetState = self.get_target_pos()
        # Only get end effector position
        robotObs, _ = self.robot.get_observation()
        robotPos = robotObs[:3]

        # Compute reward
        # reward_state goes from 0 to 1
        # is 1 if success
        reward_state = targetState
        info = {"reward_state": targetState}
        if(self.sparse_reward):
            if(success):
                reward = self.reward_success
            else:
                reward = 0
        else:
            reward_near = - np.linalg.norm(targetWorldPos - robotPos)

            reward = reward_near + reward_state
            info = {"reward_state": reward_state, "reward_near": reward_near}
        return reward, info

    def _termination(self):
        targetPos, targetState = self.get_target_pos()
        if self.task == "slide":
            done = targetState > 0.7
        elif self.task == "hinge":
            done = targetState > 0.75
        elif self.task == "drawer":
            done = targetState > 0.8
        else:  # Task == pickup
            done = targetState > 0
        return done

    def get_target_pos(self):
        if self.task == "slide":
            link_id = self.scene.get_info()['fixed_objects']['table']['uid']
            targetWorldPos = self.p.getLinkState(link_id, 2,
                                                 physicsClientId=self.cid)[0]
            targetState = self.p.getJointState(link_id, 2,
                                               physicsClientId=self.cid)[0]

            # only keep x dim
            targetWorldPos = [targetWorldPos[0] - 0.1, 0.75, 0.74]
            targetState = self._normalize(targetState, 0, 0.56)
        elif self.task == "hinge":
            link_id = \
                self.scene.get_info()['fixed_objects']['hinged_drawer']['uid']
            targetWorldPos = self.p.getLinkState(link_id, 1,
                                                 physicsClientId=self.cid)[0]
            targetState = self.p.getJointState(link_id, 1,
                                               physicsClientId=self.cid)[0]

            targetWorldPos = [targetWorldPos[0] + 0.02, targetWorldPos[1], 1]
            # table is id 0,
            # hinge door state(0) increases as it moves to left 0 to 1.74
            targetState = self.p.getJointState(0, 0,
                                               physicsClientId=self.cid)[0]
            targetState = self._normalize(targetState, 0, 1.74)
        elif self.task == "drawer":  # self.task == "drawer":
            link_id = self.scene.get_info()['fixed_objects'][self.task]['uid']
            targetWorldPos = self.p.getLinkState(link_id, 0,
                                                 physicsClientId=self.cid)[0]
            targetState = self.p.getJointState(link_id, 0,
                                               physicsClientId=self.cid)[0]
            targetWorldPos = [-0.05, targetWorldPos[1] - 0.41, 0.53]
            # self.p.addUserDebugText("O", textPosition=targetWorldPos, textColorRGB=[0, 0, 1])
            targetState = self._normalize(targetState, 0, 0.23)
        else:
            lifted = False
            for name in self.scene.table_objs:
                target_obj = \
                    self.scene.get_info()['movable_objects'][name]
                base_pos = p.getBasePositionAndOrientation(
                                target_obj["uid"],
                                physicsClientId=self.cid)[0]
                # if(p.getNumJoints(target_obj["uid"]) == 0):
                #     pos = base_pos
                # else:
                #     pos = p.getLinkState(target_obj["uid"], 0)[0]

                # self.p.addUserDebugText("O", textPosition=pos,
                #                         textColorRGB=[0, 0, 1])
                # 2.5cm above initial position and object not already in box
                if(base_pos[-1] >= target_obj["initial_pos"][-1] + 0.020
                   and not self.obj_in_box(name)):
                    lifted = True
            targetState = lifted
            # Return position of current target for training
            curr_target_uid = \
                self.scene.get_info()['movable_objects'][self.target]["uid"]
            if(p.getNumJoints(curr_target_uid) == 0):
                targetWorldPos = p.getBasePositionAndOrientation(
                    curr_target_uid,
                    physicsClientId=self.cid)[0]
            else:
                targetWorldPos = p.getLinkState(curr_target_uid, 0)[0]
        return targetWorldPos, targetState  # normalized

    def move_to_target(self, target_pos):
        tcp_pos = self.get_obs()["robot_obs"][:3]
        # To never collide with the box
        z_value = max(target_pos[2] + 0.09, 0.8)
        up_target = [tcp_pos[0],
                     tcp_pos[1],
                     z_value]
        initial_orn = self.start_orn.copy()
        # Move up from starting pose
        a = [up_target, initial_orn, 1]
        tcp_pos = self.move_to(tcp_pos, a)

        # Move in xy
        reach_target = [*target_pos[:2], tcp_pos[-1]]
        a = [reach_target, initial_orn, 1]
        tcp_pos = self.move_to(tcp_pos, a)

        # Offset relative to gripper frame
        tcp_mat = pos_orn_to_matrix(target_pos, initial_orn)
        offset_global_frame = tcp_mat @ self.offset
        move_to = offset_global_frame[:3]

        # Move to target
        a = [move_to, initial_orn, 1]
        tcp_pos = self.move_to(tcp_pos, a)

    def move_to(self, curr_pos, action):
        # action = [pos, orn, gripper_action]
        target = action[0]
        # env.robot.apply_action(a)
        last_pos = target
        # When robot is moving and far from target
        while(np.linalg.norm(curr_pos - target) > 0.01
              and np.linalg.norm(last_pos - curr_pos) > 0.001):
            last_pos = curr_pos
            self.robot.apply_action(action)
            for i in range(self.action_repeat):
                self.p.stepSimulation(physicsClientId=self.cid)
                self.scene.step()
            curr_obs = self.get_obs()
            self.save_and_viz_obs(curr_obs)
            curr_pos = curr_obs["robot_obs"][:3]
        return curr_pos

    def save_and_viz_obs(self, obs):
        if(self.viz):
            for cam_name, _ in self.cam_ids.items():
                cv2.imshow("%s_cam" % cam_name,
                           obs['rgb_obs']["rgb_%s" % cam_name][:, :, ::-1])
            cv2.waitKey(1)
        if(self.save_images):
            for cam_name, _ in self.cam_ids.items():
                os.makedirs('./images/%s_orig' % cam_name, exist_ok=True)
                cv2.imwrite("./images/%s_orig/img_%04d.png"
                            % (cam_name, self.obs_it),
                            obs['rgb_obs']["rgb_%s" % cam_name][:, :, ::-1])
        self.obs_it += 1

    def move_to_box(self, sample=False):
        # Box does not move
        r_obs = self.get_obs()["robot_obs"]
        tcp_pos, _ = r_obs[:3], r_obs[3:6]
        if(sample):
            # rand_sample over 80% of space
            top_left, bott_right = self.box_3D_end_points
            w, h, _ = np.abs((np.array(top_left) - bott_right)/2) * 0.8
            x_pos = np.random.uniform(top_left[0] + w, bott_right[0] - w)
            y_pos = np.random.uniform(top_left[1] - h, bott_right[1] + h)
        else:
            center_x, center_y, z = self.box_pos[:3]
            x_pos = center_x
            y_pos = center_y
        box_pos = [x_pos, y_pos, 0.65]

        # Move up
        initial_orn = self.start_orn.copy()
        up_target = [*tcp_pos[:2], box_pos[2] + 0.2]
        a = [up_target, initial_orn, -1]  # -1 means closed
        tcp_pos = self.move_to(tcp_pos, a)

        # Move to obj up
        up_target = [*box_pos[:2], tcp_pos[-1]]
        a = [up_target, initial_orn, -1]  # -1 means closed
        tcp_pos = self.move_to(tcp_pos, a)

        # Move down
        box_pos = [*box_pos[:2], tcp_pos[-1] - 0.12]
        a = [box_pos, initial_orn, -1]  # -1 means closed
        tcp_pos = self.get_obs()["robot_obs"][:3]
        self.move_to(tcp_pos, a)

        # Get new position and orientation
        # pos, z angle, action = open gripper
        tcp_pos = self.get_obs()["robot_obs"][:3]
        a = [tcp_pos, initial_orn, 1]  # drop object
        for i in range(8):
            curr_obs = self.get_obs()
            self.save_and_viz_obs(curr_obs)
            self.robot.apply_action(a)
            for i in range(self.action_repeat):
                self.p.stepSimulation(physicsClientId=self.cid)
                self.scene.step()
        return self.obj_in_box(self.target)

    # Success check
    def obj_in_box(self, obj_name):
        box_pos = self.box_pos
        obj_uid = self.scene.get_info()['movable_objects'][obj_name]['uid']
        targetPos = p.getBasePositionAndOrientation(
            obj_uid,
            physicsClientId=self.cid)[0]
        # x range
        x_range, y_range = False, False
        if(targetPos[0] > box_pos[0] - 0.12 and
           targetPos[0] <= box_pos[0] + 0.12):
            x_range = True
        if(targetPos[1] > box_pos[1] - 0.2 and
           targetPos[1] <= box_pos[1] + 0.2):
            y_range = True
        return x_range and y_range

    def all_objs_in_box(self):
        for obj_name in self.scene.table_objs:
            if(not self.obj_in_box(obj_name)):
                return False
        return True

    def check_success(self, any=False):
        self.move_to_box()
        if(any):
            success = False
            for name in self.scene.table_objs:
                if(self.obj_in_box(name)):
                    success = True
        else:
            success = self.obj_in_box(self.target)
        return success

    # DEBUG
    def _printJoints(self):
        for i in range(self.object_num):
            print("object%d" % i)
            for j in range(self.p.getNumJoints(i, physicsClientId=self.cid)):
                joint_info = self.p.getJointInfo(i, j,
                                                 physicsClientId=self.cid)
                print(joint_info)

    def step_debug(self):
        # ######### Debug link positions ################
        for i in range(self.p.getNumJoints(0)):  # 0 is table
            s = self.p.getLinkState(0, i)
            # print(p.getJointInfo(0,i))
            self.p.addUserDebugText(str(i),
                                    textPosition=s[0],
                                    textColorRGB=[1, 0, 0])
        # ######### Debug link positions ################
        self.p.stepSimulation()
