import math
import sys

import numpy as np

from vapo.affordance.utils.utils import get_transforms
from vapo.agent.core.sac import SAC
from vapo.agent.core.target_search import TargetSearch
from vapo.agent.core.utils import tt


class VAPOAgent(SAC):
    def __init__(self, cfg, sac_cfg=None, wandb_login=None, resume=False):
        super(VAPOAgent, self).__init__(**sac_cfg, wandb_login=wandb_login, resume=resume)
        _cam_id = self._find_cam_id()
        _aff_transforms = get_transforms(cfg.affordance.transforms.validation, cfg.target_search.aff_cfg.img_size)

        # initial pose
        _initial_obs = self.env.get_obs()["robot_obs"]
        self.origin = _initial_obs[:3]

        # To enumerate static cam preds on target search
        self.no_detected_target = 0

        args = {"cam_id": _cam_id, "initial_pos": self.origin, "aff_transforms": _aff_transforms, **cfg.target_search}
        _class_label = self.get_task_label()
        self.target_search = TargetSearch(self.env, class_label=_class_label, **args)

        # Target specifics
        self.env.target_search = self.target_search
        self.env.curr_detected_obj, _ = self.target_search.compute(rand_sample=True)
        self.eval_env = self.env
        self.radius = self.env.termination_radius  # Distance in meters
        self.sim = True

    def get_task_label(self):
        task = self.env.task
        if task == "hinge":
            return 1
        elif task == "drawer":
            return 2
        elif task == "slide":
            return 3
        else:  # pickup
            return None

    def _find_cam_id(self):
        for i, cam in enumerate(self.env.cameras):
            if "static" in cam.name:
                return i
        return 0

    # Model based methods
    def detect_and_correct(self, env, obs, noisy=False, rand_sample=True):
        if obs is None:
            obs = env.reset()
        # Compute target in case it moved
        # Area center is the target position + 5cm in z direction
        env.move_to_target(self.origin)
        target_pos, no_target = self.target_search.compute(env, noisy=noisy, rand_sample=rand_sample)
        if no_target:
            self.no_detected_target += 1
        res = self.correct_position(env, obs, target_pos, no_target)
        return res

    def correct_position(self, env, s, target_pos, no_target):
        # Set current_target in each episode
        env.curr_detected_obj = target_pos
        env.move_to_target(target_pos)
        # as we moved robot, need to update target and obs
        # for rl policy
        return env, env.observation(env.get_obs()), no_target

    # RL Policy
    def learn(
        self, total_timesteps=10000, log_interval=100, full_eval_interval=200, max_episode_length=None, n_eval_ep=5
    ):
        if not isinstance(total_timesteps, int):  # auto
            total_timesteps = int(total_timesteps)
        episode_return, episode_length = 0, 0
        if max_episode_length is None:
            max_episode_length = sys.maxsize  # "infinite"

        # plot_data = {"actor_loss": [], "critic_loss": [], "ent_coef_loss": [], "ent_coef": []}

        _log_n_ep = log_interval // max_episode_length
        _full_eval_interval = full_eval_interval // max_episode_length
        if _log_n_ep < 1:
            _log_n_ep = 1

        # Move to target position only one
        # Episode ends if outside of radius
        self.env, s, _ = self.detect_and_correct(self.env, None, noisy=True)
        for ts in range(1, total_timesteps + 1):
            s, done, success, episode_return, episode_length, plot_data, info = self.training_step(
                s, self.curr_ts, episode_return, episode_length
            )

            # End episode
            timeout = max_episode_length and (episode_length >= max_episode_length)
            end_ep = timeout or done

            # Log interval (sac)
            if (ts % log_interval == 0 and not self._log_by_episodes) or (
                self._log_by_episodes and end_ep and self.episode % _log_n_ep == 0
            ):
                self.best_eval_return, self.most_tasks = self._eval_and_log(
                    self.curr_ts, self.episode, self.most_tasks, self.best_eval_return, n_eval_ep, max_episode_length
                )

            eval_all_objs = self.episode % _full_eval_interval == 0
            if (ts % full_eval_interval == 0 and not self._log_by_episodes) or (
                self._log_by_episodes and end_ep and eval_all_objs
            ):
                if self.eval_env.rand_positions and eval_all_objs:
                    _, self.most_full_tasks = self._eval_and_log(
                        self.curr_ts,
                        self.episode,
                        self.most_full_tasks,
                        self.best_eval_return,
                        n_eval_ep,
                        max_episode_length,
                        eval_all_objs=True,
                    )
            if end_ep:
                self.best_return = self._on_train_ep_end(
                    self.curr_ts,
                    self.episode,
                    total_timesteps,
                    self.best_return,
                    episode_length,
                    episode_return,
                    success,
                    plot_data,
                )
                # Reset everything
                self.episode += 1
                self.env.obs_it = 0
                episode_return, episode_length = 0, 0
                self.env, s, _ = self.detect_and_correct(self.env, None, noisy=True)
            self.curr_ts += 1
        # Evaluate at end of training
        for eval_all_objs in [False, True]:
            if eval_all_objs and self.env.rand_positions or not eval_all_objs:
                self.best_eval_return, most_tasks = self._eval_and_log(
                    self.curr_ts,
                    self.episode,
                    self.most_tasks,
                    self.best_eval_return,
                    n_eval_ep,
                    max_episode_length,
                    eval_all_objs=eval_all_objs,
                )

    # Only applies to pickup task
    def eval_all_objs(
        self, env, max_episode_length=100, n_episodes=None, print_all_episodes=False, render=False, save_images=False
    ):
        if env.rand_positions is None:
            return

        # Store current objs to restore them later
        previous_objs = env.scene.table_objs

        #
        tasks = env.scene.obj_names
        n_objs = len(env.scene.rand_positions)
        n_total_objs = len(tasks)
        objs_success = {}
        episodes_success = []
        for i in range(math.ceil(n_total_objs / n_objs)):
            if len(tasks[i:]) >= n_objs:
                curr_objs = tasks[i * n_objs : i * n_objs + n_objs]
            else:
                curr_objs = tasks[i:]
            env.get_scene_with_objects(curr_objs)
            mean_reward, mean_length, ep_success, success_count = self.evaluate(
                env, max_episode_length=max_episode_length
            )
            objs_success.update(success_count)
            episodes_success.extend(ep_success)

        self.log.info(
            "Full evaluation over %d objs \n" % n_total_objs
            + "Success: %d/%d " % (np.sum(episodes_success), len(episodes_success))
        )

        # Restore scene before loading full sweep eval
        env.get_scene_with_objects(previous_objs)
        return episodes_success, objs_success

    def evaluate(self, env, max_episode_length=150, n_episodes=5, print_all_episodes=True):
        ep_returns, ep_lengths = [], []
        tasks, task_it = [], 0

        if env.task == "pickup":
            if self.target_search.mode == "env":
                tasks = env.scene.table_objs
                n_episodes = len(tasks)
            else:
                # Keep same scene that with wich was trained
                env.reset(eval=True)
                target_pos, no_target, center_targets = self.target_search.compute(env, return_all_centers=True)
                n_episodes = len(center_targets)

        ep_success = []
        success_objs = {}
        # One episode per task
        for episode in range(n_episodes):
            s = env.reset(eval=True)
            # rand_pos = self.origin + np.random.uniform(low=[-0.5, -0.01, -0.02],
            #                                            high=[0.0, 0.01, 0.05],
            #                                            size=(len(self.origin)))

            # env.move_to_target(rand_pos)
            if env.task == "pickup":
                if self.target_search.mode == "env":
                    env.target = tasks[task_it]
                    target_pos, no_target = self.target_search.compute(env, rand_sample=False, noisy=False)
                else:
                    target_pos = center_targets[task_it]["target_pos"]
                    env.target = center_targets[task_it]["target_str"]
                task_it += 1
            else:
                target_pos, no_target = self.target_search.compute(env)
            episode_length, episode_return = 0, 0
            done = False
            # Correct Position
            env, s, _ = self.correct_position(env, s, target_pos, no_target)
            while episode_length < max_episode_length and not done:
                # sample action and scale it to action space
                s = env.transform_obs(tt(s), "validation")
                a, _ = self._pi.act(s, deterministic=True)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                s = ns
                episode_return += r
                episode_length += 1
            success_objs[env.target] = info["success"]
            ep_success.append(info["success"])
            ep_returns.append(episode_return)
            ep_lengths.append(episode_length)
            if print_all_episodes:
                print_str = (
                    "EVALUATION [%d] %s, " % (episode, env.target)
                    + "Return: %.3f, " % (episode_return)
                    + "Success: %s, " % str(info["success"])
                    + "Steps: %d" % episode_length
                )
                self.log.info(print_str)

        # mean and print
        mean_reward, reward_std = np.mean(ep_returns), np.std(ep_returns)
        mean_length, length_std = np.mean(ep_lengths), np.std(ep_lengths)

        self.log.info(
            "Success: %d/%d " % (np.sum(ep_success), len(ep_success))
            + "Mean return: %.3f +/- %.3f, " % (mean_reward, reward_std)
            + "Mean length: %.3f +/- %.3f, over %d episodes" % (mean_length, length_std, n_episodes)
        )
        return mean_reward, mean_length, ep_success, success_objs

    # Only applies to tabletop
    def tidy_up(self, env, max_episode_length=100):
        tasks = []
        # get from static cam affordance
        if env.task == "pickup":
            tasks = self.env.scene.table_objs
            n_tasks = len(tasks)

        ep_success = []
        total_ts = 0
        env.reset()
        # Set total timeout to timeout per task times all tasks + 1
        while (
            total_ts <= max_episode_length * n_tasks and self.no_detected_target < 3 and not self.env.all_objs_in_box()
        ):
            episode_length, episode_return = 0, 0
            done = False
            # Search affordances and correct position:
            env, s, no_target = self.detect_and_correct(env, self.env.get_obs(), rand_sample=True)
            if no_target:
                # If no target model will move to initial position.
                # Search affordance from this position again
                env, s, no_target = self.detect_and_correct(env, self.env.get_obs(), rand_sample=True)

            # If it did not find a target again, terminate everything
            while episode_length < max_episode_length and self.no_detected_target < 3 and not done:
                # sample action and scale it to action space
                s = env.transform_obs(tt(s), "validation")
                a, _ = self._pi.act(s, deterministic=True)
                a = a.cpu().detach().numpy()
                ns, r, done, info = env.step(a)
                s = ns
                episode_return += r
                episode_length += 1
                total_ts += 1
                success = info["success"]
            ep_success.append(success)
            env.episode += 1
            env.obs_it = 0
        self.log.info("Success: %d/%d " % (np.sum(ep_success), len(ep_success)))
        return ep_success
