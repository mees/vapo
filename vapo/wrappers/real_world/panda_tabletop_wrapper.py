import logging
import math
import time

import gym
import numpy as np
from robot_io.utils.utils import quat_to_euler

from vapo.utils.utils import get_3D_end_points

log = logging.getLogger(__name__)

WAIT_AFTER_GRIPPER_CLOSE = 1
WAIT_FOR_WIDTH_MEASUREMENT = 1
MOVE_UP_AFTER_GRASP = 0.03
GRIPPER_SUCCESS_THRESHOLD = 0.01
GRIPPER_WIDTH_FAIL = 0.075


class PandaEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        d_pos,
        d_rot,
        gripper_success_threshold,
        reward_fail,
        reward_success,
        termination_radius,
        offset,
        *args,
        **kwargs,
    ):
        super().__init__(env)
        self.d_pos = d_pos
        self.d_rot = d_rot
        self.gripper_success_threshold = gripper_success_threshold
        self.reward_fail = reward_fail
        self.reward_success = reward_success
        self.termination_radius = termination_radius
        self.target_pos = None
        self.offset = offset["pickup"]
        self.task = "pickup"
        self.start_orn = np.array([math.pi, 0, 0])
        if "box_pos" in kwargs:
            self.box_pos = kwargs["box_pos"]
            self.box_3D_end_points = get_3D_end_points(*self.box_pos, *kwargs["box_dims"])

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    def reset(self, target_pos=None, target_orn=None):
        self.env.robot.open_gripper()
        if target_pos is not None and target_orn is not None:
            self.target_pos = target_pos
            move_to = target_pos + self.offset[self.task]
        else:
            move_to = target_pos
        return self.transform_obs(self.env.reset(move_to, target_orn))

    def check_success(self):
        time.sleep(WAIT_AFTER_GRIPPER_CLOSE)
        pos, orn = self.env.robot.get_tcp_pos_orn()
        pos[2] += MOVE_UP_AFTER_GRASP
        self.env.robot.move_async_cart_pos_abs_lin(pos, orn)
        time.sleep(WAIT_FOR_WIDTH_MEASUREMENT)
        gripper_width = self.env.robot.get_state()["gripper_opening_width"]
        if gripper_width > GRIPPER_WIDTH_FAIL:
            log.error("Gripper action seems to have no effect.")
            raise Exception
        return gripper_width > self.gripper_success_threshold

    def move_to_box(self):
        box_pos = self.box_pos
        pos, orn = self.env.robot.get_tcp_pos_orn()
        pos[2] = box_pos[-1] + MOVE_UP_AFTER_GRASP * 2
        self.env.robot.move_cart_pos_abs_ptp(pos, orn)
        time.sleep(WAIT_FOR_WIDTH_MEASUREMENT)
        self.env.robot.move_cart_pos_abs_ptp(box_pos, orn)
        self.env.robot.open_gripper(blocking=True)

    def put_back_object(self):
        pos, orn = self.env.robot.get_tcp_pos_orn()
        pos[2] -= MOVE_UP_AFTER_GRASP * 0.8
        self.env.robot.move_async_cart_pos_abs_lin(pos, orn)
        self.env.robot.open_gripper(blocking=True)

    def check_termination(self, current_pos):
        return np.linalg.norm(self.target_pos - current_pos) > self.termination_radius

    def step(self, action, move_to_box=False):
        assert len(action) == 5

        rel_target_pos = np.array(action[:3]) * self.d_pos
        rel_target_orn = np.array([0, 0, action[3]]) * self.d_rot
        gripper_action = action[-1]

        curr_pos = self.env.robot.get_tcp_pos_orn()[0]
        depth_thresh = curr_pos[-1] <= self.env.workspace_limits[0][-1] + 0.01
        if depth_thresh:
            print("depth tresh")
            gripper_action = -1

        action = {"motion": (rel_target_pos, rel_target_orn, gripper_action), "ref": "rel"}

        obs, reward, done, info = self.env.step(action)

        info["success"] = False

        if gripper_action == -1:
            done = True
            if self.check_success():
                reward = self.reward_success
                info["success"] = True
                if move_to_box:
                    self.move_to_box()
                else:
                    self.put_back_object()
            else:
                reward = self.reward_fail
                info["failure_case"] = "failed_grasp"
        else:
            done = self.check_termination(obs["robot_state"]["tcp_pos"])
            if done:
                reward = self.reward_fail
                info["failure_case"] = "outside_radius"
        if "failure_case" in info:
            print(info["failure_case"])

        obs = self.transform_obs(obs)
        return obs, reward, done, info

    def get_obs(self):
        return self.transform_obs(self.env._get_obs())

    @staticmethod
    def transform_obs(obs):
        robot_obs = obs["robot_state"]
        obs["robot_obs"] = np.concatenate(
            [robot_obs["tcp_pos"], quat_to_euler(robot_obs["tcp_orn"])[-1:], [robot_obs["gripper_opening_width"]]]
        )
        del obs["robot_state"]
        return obs
