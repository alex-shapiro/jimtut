from typing import Never, final, override

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from torch._prims_common import DeviceLikeType
from torch.distributions.multivariate_normal import MultivariateNormal

from replay_buffer import ReplayBuffer
from torch import Tensor, nn


@final
class IdpAgent:
    """Agent for controlling an inverted double pendulum"""

    def __init__(self, seed: int = 2025):
        # sim environment
        self.env: gym.Env[np.ndarray, float] = gym.make("InvertedDoublePendulum-v5")  # pyright: ignore[reportUnknownMemberType]
        state_space: Box = self.env.observation_space  # pyright: ignore[reportAssignmentType, reportUnknownMemberType]
        action_space: Box = self.env.action_space  # pyright: ignore[reportAssignmentType, reportUnknownMemberType]

        # RNG seeds
        np.random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.env.action_space.seed(seed)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.seed = seed

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100_000,
            state_space=state_space,
            action_space=action_space,
        )

        # hyperparameters
        self.num_cells = 256
        self.lr = 3e-4
        self.max_grad_norm = 1.0
        self.batch_size = 1000
        self.n_frames = 50_000

        # PPO parameters
        self.sub_batch_size = 64
        self.n_epochs = 10
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.lmbda = 0.95
        self.entropy_eps = 1e-4

        # models
        self.device = torch.device("cpu")

    def train(self):
        self.check_env_specs()

    def check_env_specs(self):
        print(f"observation space: {self.env.observation_space}")
        print(f"action space: {self.env.action_space}")
        print(f"env spec: {self.env.spec}")


@final
class ActorCritic(nn.Module):
    """Actor-Critic neural net"""

    def __init__(
        self,
        d_state: int,
        d_action: int,
        initial_action_std: float,
        device: DeviceLikeType,
        d_hidden: int = 64,
    ):
        super(ActorCritic, self).__init__()  # pyright: ignore[reportUnknownMemberType]
        self.d_action = d_action
        self.action_var = torch.full(
            size=[d_action],
            fill_value=initial_action_std**2,
        ).to()
        self.actor = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_action),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    @override
    def forward(self) -> Never:
        raise NotImplementedError

    def act(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        action_mean: Tensor = self.actor(state)  # pyright: ignore[reportAny]
        cov_matrix = self.action_var.diag().unsqueeze(dim=0)
        dist = MultivariateNormal(loc=action_mean, covariance_matrix=cov_matrix)
        action = dist.sample().detach()
        action_logprob: Tensor = dist.log_prob(action).detach()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        state_val: Tensor = self.critic(state).detach()  # pyright: ignore[reportAny]
        return (action, action_logprob, state_val)  # pyright: ignore[reportUnknownVariableType]

    def evaluate(self, state: Tensor, action: Tensor) -> Tensor:
        action_mean: Tensor = self.actor(state)  # pyright: ignore[reportAny]
        action_var = self.action_var.expand_as(action_mean)
        cov_matrix: Tensor = action_var.diag_embed().to(self.device)
        dist = MultivariateNormal(loc=action_mean, covariance_matrix=cov_matrix)
        action_logprobs = dist.log_prob(action).detach()
        dist_entropy = dist.entropy()


if __name__ == "__main__":
    agent = IdpAgent()
    agent.train()
