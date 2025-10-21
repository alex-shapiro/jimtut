from typing import final, override

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch import Tensor, nn

from replay_buffer import ReplayBuffer


@final
class AcrobotAgent:
    def __init__(self, seed: int = 2025):
        # sim environment
        self.env: gym.Env[np.ndarray, int] = gym.make("Acrobot-v1")  # pyright: ignore[reportUnknownMemberType]
        self.eval_env: gym.Env[np.ndarray, int] = gym.make("Acrobot-v1")  # pyright: ignore[reportUnknownMemberType]
        state_space: Box = self.env.observation_space  # pyright: ignore[reportAssignmentType]
        action_space: Discrete = self.env.action_space  # pyright: ignore[reportAssignmentType]

        # init random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.seed = seed

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100_000,
            state_space=state_space,
            action_space=action_space,
        )

        # exploration hyperparameters
        self.num_steps = 80_000
        self.eps_initial = 1.0
        self.eps_final = 0.04
        self.exploration_fraction = 0.1

        # other hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 1e-3
        self.batch_size = 64

        # models
        self.device = torch.device("cpu")


@final
class QNet(nn.Module):
    """NN for Q-value prediction"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(QNet, self).__init__()  # pyright: ignore[reportUnknownMemberType]
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_size),
        )

    @override
    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # pyright: ignore[reportAny]
