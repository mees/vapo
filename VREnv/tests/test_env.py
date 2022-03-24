import math

import cv2
import hydra
import pybullet as p
from vr_env.envs.play_table_env import PlayTableSimEnv


def set_init_pos(task, init_pos):
    if task == "slide" or task == "drawer":
        init_pos = [
            -1.1686195081948965,
            1.5165126497924815,
            1.7042540963745911,
            -1.6031852712241403,
            -2.5717679087567484,
            2.331416872629473,
            -1.3006358472301627,
        ]
    elif task == "hinge":
        init_pos = [
            -0.3803066514807313,
            0.931053115322005,
            1.1668869976984892,
            -0.8602164833917604,
            -1.4818301463768684,
            2.78299286093898,
            -1.7318962831826747,
        ]
    elif task == "tabletop":
        init_pos = [
            0.2584550602550528,
            1.1556688342206756,
            1.145914956800992,
            -0.45965789854738986,
            -1.0171025924903545,
            1.3827150843394707,
            -1.6732613980921893,
        ]
    return init_pos


@hydra.main(config_path="../conf", config_name="config_rl")
def main(cfg):
    # init_pos = cfg.env.robot_cfg.initial_joint_positions
    # init_pos = set_init_pos(cfg.task, init_pos)
    # cfg.env.robot_cfg.initial_joint_positions = init_pos

    env = PlayTableSimEnv(**cfg.env)
    # env = PlayTableSimEnv(**cfg.env)
    p.resetDebugVisualizerCamera(cameraDistance=1.4, cameraYaw=-45, cameraPitch=-45, cameraTargetPosition=[-1, 0, 0.5])
    # Cartesian position of the gripper + joint variable for both fingers
    # fixed_angle = [- math.pi / 2, math.pi / 2, 0]
    # fixed_angle = [- math.pi, math.pi / 6 , - math.pi / 2]
    # action = ([0.40, 0.6, 0.76], fixed_angle, [1])  # slide
    # action = ([0, 0.9, 0.76], fixed_angle, [1])  # slide end
    # action = ([0.6, 0.6, 0.6], fixed_angle, [1])

    # action = ([0.15 , 0.75, 1.0], fixed_angle, [1])  # hinge
    # action = ([0.03 , 0.52, 1], fixed_angle, [1])  # hinge middle
    # action = ([-0.35 , 0.39, 1], fixed_angle, [1])  # hinge end

    # Drawer
    # fixed_angle = [- math.pi, 0, - math.pi / 2]
    # action = ([-0.05 , 0.46, 0.57], fixed_angle , 1)  # drawer start
    # action = ([-0.05 , 0.2, 0.469], fixed_angle , 1)  # drawer end

    # Pickup
    fixed_angle = [math.pi, 0, math.pi / 2]
    # action = ([-0.25, -0.25, 0.6], fixed_angle, [1])
    # action = ([-0.13, 0.9, 0.62], fixed_angle, [1])
    action = ([0.6, 0.6, 0.7], fixed_angle, [1])

    for i in range(3):  # ep
        for i in range(3000):  # ep_len
            ns, r, d, info = env.step(action)
            for i, (name, img) in enumerate(ns["rgb_obs"].items()):
                cv2.imshow("cam%d" % i, img[:, :, ::-1])
                cv2.waitKey(1)
            # pos = env.get_target_pos()[0]
            # robot_pos = ns['robot_obs'][:3]
            p.addUserDebugText("t", textPosition=action[0], textColorRGB=[1, 0, 0])
        env.reset()


if __name__ == "__main__":
    main()
