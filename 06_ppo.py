from typing import final

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal


@final
class TrajectoryBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: list[int],
        action_shape: list[int],
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


def ppo(env: gym.Env):
    pass


class CategoricalActor(nn.Module):
    def __init__(
        self,
        d_state: int,
        d_hidden: int,
        d_action: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.logits_net = (
            nn.Sequential(
                nn.Linear(d_state, d_hidden),
                activation(),
                nn.Linear(d_hidden, d_hidden),
                activation(),
                nn.Linear(d_hidden, d_action),
            ),
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        policy = self._policy(states)
        logp_actions = policy.log_prob(actions) if actions else None
        return policy, logp_actions

    def _policy(self, states: Tensor) -> Categorical:
        logits = self.logits_net(states)
        return Categorical(logits=logits)

    def _logprob(self, policy: Categorical, action: Tensor) -> Tensor:
        return policy.log_prob(action)


class GaussianActor(nn.Module):
    def __init__(
        self,
        d_state: int,
        d_hidden: int,
        d_action: int,
        activation: nn.Module,
    ):
        super().__init__()
        log_std = -0.5 * np.ones(d_action, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            activation(),
            nn.Linear(d_hidden, d_hidden),
            activation(),
            nn.Linear(d_hidden, d_action),
        )

    def forward(
        self, states: Tensor, actions: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        policy = self._policy(states)
        logp_actions = self._logprob(policy, actions) if actions else None
        return policy, logp_actions

    def _policy(self, states: Tensor) -> Normal:
        mu = self.mu_net(states)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _logprob(self, policy: Normal, action: Tensor) -> Tensor:
        return policy.log_prob(action).sum(axis=-1)


class Critic(nn.Module):
    def __init__(self, d_state: int, d_hidden: int, activation: nn.Module):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            activation,
            nn.Linear(d_hidden, d_hidden),
            activation,
            nn.Linear(d_hidden, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.value_net(states).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_space: Box,
        action_space: Box | Discrete,
        d_hidden: int = 64,
        activation: nn.Module = nn.Tanh,
    ):
        super().__init__()
        if isinstance(action_space, Box):
            self.pi = GaussianActor(
                d_space=state_space.shape[0],
                d_action=state_space.shape[0],
                d_hidden=d_hidden,
                activation=activation,
            )
        else:
            self.pi = CategoricalActor(
                d_space=state_space.shape[0],
                d_action=state_space.n,
                d_hidden=d_hidden,
                activation=activation,
            )
        self.v = Critic(
            d_space=state_space.shape[0],
            d_hidden=d_hidden,
            activation=activation,
        )

    def step(self, state: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            policy = self.pi._policy(state)
            action = policy.sample()
            logp_action = self.pi._logprob(policy, action)
            v = self.v(state)
            return (action.numpy(), logp_action.numpy(), v.numpy())

    def act(self, state: Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.pi._policy(state).sample().numpy()


def count_parameters(module: nn.Module) -> int:
    """Computes the total number of parameters in a NN module"""
    return sum(np.prod(p.shape) for p in module.parameters())


def discount_cumulative_sum(x: np.ndarray, discount: float) -> float:
    """Compute discounted cumulative sum of vectors"""
    pass


if __name__ == "__main__":
    agent = PPOAgent()
    agent.train()
    agent.evaluate(n_episodes=3, visible=True)
