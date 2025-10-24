from typing import final

import gymnasium as gym
import numpy as np
import torch


@final
class TrajectoryBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: [int],
        action_shape: [int],
        gamma: float = 0.99,
        lamda: float = 0.95,
    ):
        self.states = np.zeros([capacity, *state_shape], dtype=np.float32)
        self.actions = np.zeros([capacity, *action_shape], dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.logps = np.zeros(capacity, dtype=np.float32)
        self.gamma = gamma
        self.lamda = lamda
        self.next_index = 0
        self.trajectory_start_index = 0
        self.capacity = capacity

    def push(
        self, state: np.array, action: float, reward: float, value: float, logp: float
    ):
        self.states[self.next_index] = state
        self.actions[self.next_index] = action
        self.rewards[self.next_index] = reward
        self.values[self.next_index] = value
        self.logps[self.next_index] = logp
        self.next_index += 1

    def push_trajectory_end(self, last_val: float = 0.0):
        pass


def ppo(env: gy):
    pass
