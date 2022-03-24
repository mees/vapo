from collections import deque, namedtuple
import glob
import os
from pathlib import Path

import numpy as np

from vapo.agent.core.utils import tt


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, max_size, dict_state=False, logger=None):
        self._transition = namedtuple("transition", ["state", "action", "reward", "next_state", "terminal_flag"])
        self._data = deque(maxlen=int(max_size))
        self._max_size = max_size
        self.dict_state = dict_state
        self.last_saved_idx = -1
        self.logger = logger

    def __len__(self):
        return len(self._data)

    def add_transition(self, state, action, reward, next_state, done):
        transition = self._transition(state, action, reward, next_state, done)
        self._data.append(transition)

    def sample(self, batch_size):
        batch_indices = np.random.choice(len(self._data), batch_size)
        (batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminal_flags) = zip(
            *[self._data[i] for i in batch_indices]
        )

        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards)
        batch_terminal_flags = np.array(batch_terminal_flags).astype("uint8")
        if self.dict_state:
            v = {k: np.array([dic[k] for dic in batch_states]) for k in batch_states[0]}
            batch_states = v
            v = {k: np.array([dic[k] for dic in batch_next_states]) for k in batch_next_states[0]}
            batch_next_states = v

        return tt(batch_states), tt(batch_actions), tt(batch_rewards), tt(batch_next_states), tt(batch_terminal_flags)

    def save(self, path="./replay_buffer"):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        num_entries = len(self._data)
        for i in range(self.last_saved_idx + 1, num_entries):
            transition = self._data[i]
            if not isinstance(transition.state, dict) or not isinstance(transition.next_state, dict):
                continue
            file_name = "%s/transition_%d.npy" % (path, i)
            np.save(
                file_name,
                {
                    "state": transition.state,
                    "action": transition.action,
                    "next_state": transition.next_state,
                    "reward": transition.reward,
                    "terminal_flag": transition.terminal_flag,
                },
            )
        if num_entries - 1 - self.last_saved_idx > 0:
            self.logger.info("Saved transitions with indices : %d - %d" % (self.last_saved_idx, i))
            self.last_saved_idx = i

    def load(self, path="./replay_buffer"):
        p = Path(path)
        if p.is_dir():
            p = p.glob("*.npy")
            files = [x for x in p if x.is_file()]
            self.logger.info("Loading replay buffer...")
            if len(files) > 0:
                for file in files:
                    data = np.load(file, allow_pickle=True).item()
                    transition = self._transition(
                        data["state"], data["action"], data["reward"], data["next_state"], data["terminal_flag"]
                    )
                    self._data.append(transition)
                self.last_saved_idx = len(files)
                self.logger.info("Replay buffer loaded successfully")
            else:
                self.logger.info("No files were found in path %s" % (path))
        else:
            self.logger.info("Path %s does not have an appropiate directory address" % (path))
