
import time

import hydra
# from robot_io.cams.realsense.realsense import Realsense
from robot_io.utils.utils import FpsController

from vapo.affordance.utils.img_utils import transform_and_predict, viz_aff_centers_preds
from vapo.affordance.utils.utils import load_from_hydra


@hydra.main(config_path="../config", config_name="real_world_teleop")
def main(cfg):

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    input_device = hydra.utils.instantiate(cfg.input, robot=robot)

    fps = FpsController(cfg.freq)

    env.reset()
    gripper_cam_aff_net, gripper_cfg = load_from_hydra(cfg.gripper_cam)
    static_cam_aff_net, static_cfg = load_from_hydra(cfg.static_cam)

    recorder = hydra.utils.instantiate(cfg.recorder)
    t1 = time.time()
    while True:
        action, record_info = input_device.get_action()
        obs, _, _, _ = env.step(action)
        g_out = transform_and_predict(gripper_cam_aff_net,
                                        gripper_cfg.dataset.transforms_cfg["validation"],
                                        obs["rgb_gripper"],
                                        gripper_cfg.img_size["gripper"],
                                    )
        centers, mask, directions, _, _ = g_out
        viz_aff_centers_preds(obs["rgb_gripper"],
                                mask,
                                directions,
                                centers,
                                cam_type="gripper")
        s_out = transform_and_predict(static_cam_aff_net,
                                      static_cfg.dataset.transforms_cfg["validation"],
                                      obs["rgb_static"],
                                      static_cfg.img_size["static"],
                                )
        centers, mask, directions, _, _ = s_out
        viz_aff_centers_preds(obs["rgb_static"],
                                mask,
                                directions,
                                centers,
                                cam_type="static")
        robot.visualize_joint_states()
        recorder.step(action, obs, record_info)
        env.render()
        fps.step()
        print(1 / (time.time() - t1))
        t1 = time.time()


if __name__ == "__main__":
    main()
