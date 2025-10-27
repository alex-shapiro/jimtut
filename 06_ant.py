# Mujoco Ant agent using PPO (proximal policy optimization)

import typing
from dataclasses import dataclass
from typing import final

import gymnasium as gym
import numpy as np
import torch
from torch.optim import AdamW
from gymnasium.spaces import Box, Discrete
from torch import Tensor, nn
from torch.distributions import Categorical, Normal


@final
class AntAgent:
    def __init__(
        self,
        seed: int = 2025,
        steps_per_epoch: int = 4000,
        epochs: int = 50,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        pi_lr: float = 3e-4,
        value_lr: float = 1e-3,
        train_policy_iters: int = 80,
        train_value_iters: int = 80,
        lamda: float = 0.97,
        max_episode_length: int = 1000,
        target_kl: float = 0.01,
        save_frequency: int = 10,
    ):
        super().__init__()

        self.env: gym.Env[np.ndarray, np.ndarray] = gym.make("Ant-v5") # pyright: ignore[reportUnknownMemberType]
        state_space: Box = self.env.observation_space # pyright: ignore[reportAssignmentType]
        action_space: Box = self.env.action_space # pyright: ignore[reportAssignmentType]

        # RNG seeds
        np.random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.env.action_space.seed(seed)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.seed = seed

        # trajectory buffer
        self.trajectories = TrajectoryBuffer(
            capacity=4000,
            state_shape=list(state_space.shape),
            action_shape=list(action_space.shape),
        )

        # hyperparameters
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.train_policy_iters = train_policy_iters
        self.train_value_iters = train_value_iters
        self.lamda = lamda
        self.max_episode_length = max_episode_length
        self.target_kl = target_kl
        self.save_frequency = save_frequency

        # models
        self.device = torch.device("cpu")
        self.actor_critic = ActorCritic(
            state_space=state_space,
            action_space=action_space,
        )
        self.policy_optimizer = AdamW(self.actor_critic.pi.parameters(), lr=pi_lr)
        self.value_optimizer = AdamW(self.actor_critic.v.parameters(), lr=value_lr)

    def train(self):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch}")
            state, _ = self.env.reset()
            episode_return = 0
            episode_length = 0

            for t in range(self.steps_per_epoch):
                action, logp_action, value = self.actor_critic.step(torch.as_tensor(state))
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_return += value
                episode_length += 1

                self.trajectories.push(
                    state=state,
                    action=action,
                    logp=float(logp_action),
                    value=float(value),
                    reward=float(reward),
                )

                state = next_state
                if done or truncated or (t == self.steps_per_epoch - 1):
                    _, _, value = self.actor_critic.step(torch.as_tensor(state))
                    self.trajectories.push_episode_end(value)
                    state, _ = self.env.reset()
                    episode_return = 0
                    episode_length = 0

            self.update()

    def update(self):
        batch = self.trajectories.get_batch()
        policy_loss = torch.zeros(1)
        value_loss = torch.zeros(1)
        policy_loss_old, _ = self.policy_loss(batch)
        value_loss_old = self.value_loss(batch)

        # train policy
        for i in range(self.train_policy_iters):
            self.policy_optimizer.zero_grad()
            policy_loss, policy_info = self.policy_loss(batch)
            if policy_info.approximate_kl > 1.5 * self.target_kl:
                print(f"early stopping at step {i} due to reaching max KL")
                break
            policy_loss.backward() # pyright: ignore[reportUnknownMemberType]
            self.policy_optimizer.step() # pyright: ignore[reportUnknownMemberType]

        # learn value function
        for i in range(self.train_value_iters):
            self.value_optimizer.zero_grad()
            value_loss = self.value_loss(batch)
            value_loss.backward() # pyright: ignore[reportUnknownMemberType]
            self.value_optimizer.step() # pyright: ignore[reportUnknownMemberType]

        # log changes from the update
        print(f"policy loss: {policy_loss_old}")
        print(f"value loss: {value_loss_old}")
        print(f"Δ policy loss: {policy_loss.item() - policy_loss_old}") # pyright: ignore[reportUnknownMemberType]
        print(f"Δ value loss: {value_loss.item() - value_loss_old}") # pyright: ignore[reportUnknownMemberType]

    def policy_loss(self, batch: "TrajectoryBatch") -> tuple[Tensor, "PolicyInfo"]:
        # policy loss
        pi: Normal
        logps: Tensor
        batch_logps = torch.as_tensor(batch.logps)
        pi, logps = self.actor_critic.pi(batch.states, batch.actions)
        ratio = torch.exp(logps - torch.as_tensor(batch_logps))
        clipped_adv = torch.clamp(ratio, 1 - self.clip_ratio)
        policy_loss = (torch.min(ratio * torch.as_tensor(batch.advantages), clipped_adv)).mean()

        # additional policy info
        approximate_kl = (batch_logps - logps).mean().item()
        mean_entropy = pi.entropy().mean().item()
        clipped_fraction = (
            (ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio))
            .to(torch.float32)
            .mean()
            .item()
        )
        policy_info = PolicyInfo(
            approximate_kl=approximate_kl,
            mean_entropy=mean_entropy,
            clipped_fraction=clipped_fraction,
        )

        return policy_loss, policy_info

    def value_loss(self, batch: "TrajectoryBatch") -> Tensor:
        state = torch.as_tensor(batch.states)
        return ((self.actor_critic.v(state) - batch.returns) ** 2).mean()


@final
class CategoricalActor(nn.Module):
    """Policy prediction over discrete 1D action spaces"""

    def __init__(
        self,
        d_state: int,
        d_hidden: int,
        d_action: int,
        activation: nn.Module,
    ):
        super().__init__() # pyright: ignore[reportUnknownMemberType]
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
    ) -> tuple[Categorical, Tensor | None]:
        policy = self.policy(states)
        logp_actions = typing.cast(Tensor | None, policy.log_prob(actions) if actions else None)
        return policy, logp_actions

    def policy(self, states: Tensor) -> Categorical:
        logits = typing.cast(Tensor, self.logits_net(states)) # pyright: ignore[reportCallIssue, reportUnknownVariableType]
        return Categorical(logits=logits)

    def logprob(self, policy: Categorical, action: Tensor) -> Tensor:
        return typing.cast(Tensor, policy.log_prob(action))


@final
class GaussianActor(nn.Module):
    """Policy prediction over continuous 1D action spaces"""

    def __init__(
        self,
        d_state: int,
        d_hidden: int,
        d_action: int,
        activation: nn.Module,
    ):
        super().__init__() # pyright: ignore[reportUnknownMemberType]
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
    ) -> tuple[Normal, Tensor | None]:
        policy = self.policy(states)
        logp_actions = self.logprob(policy, actions) if actions else None
        return policy, logp_actions

    def policy(self, states: Tensor) -> Normal:
        mu = self.mu_net(states)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def logprob(self, policy: Normal, action: Tensor) -> Tensor:
        return typing.cast(Tensor, policy.log_prob(action).sum(axis=-1)) # pyright: ignore[reportCallIssue]


@final
class Critic(nn.Module):
    """Value prediction over 1D spaces"""

    def __init__(self, d_state: int, d_hidden: int, activation: nn.Module):
        super().__init__() # pyright: ignore[reportUnknownMemberType]
        self.value_net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            activation,
            nn.Linear(d_hidden, d_hidden),
            activation,
            nn.Linear(d_hidden, 1),
        )

    def forward(self, states: Tensor) -> Tensor:
        return self.value_net(states).squeeze(-1)


@final
class ActorCritic(nn.Module):
    def __init__(
        self,
        state_space: Box,
        action_space: Box | Discrete,
        d_hidden: int = 64,
        activation: nn.Module = nn.Tanh, # pyright: ignore[reportArgumentType]
    ):
        super().__init__() # pyright: ignore[reportUnknownMemberType]
        if isinstance(action_space, Box):
            self.pi = GaussianActor(
                d_state=state_space.shape[0],
                d_hidden=d_hidden,
                d_action=action_space.shape[0],
                activation=activation,
            )
        else:
            self.pi = CategoricalActor(
                d_state=state_space.shape[0],
                d_action=int(action_space.n),
                d_hidden=d_hidden,
                activation=activation,
            )
        self.v = Critic(
            d_state=state_space.shape[0],
            d_hidden=d_hidden,
            activation=activation,
        )

    def step(self, state: Tensor) -> tuple[np.ndarray, np.ndarray, float]:
        with torch.no_grad():
            policy = self.pi.policy(state)
            action = policy.sample()
            logp_action = self.pi.logprob(policy, action) # pyright: ignore[reportArgumentType]
            v = float(self.v(state))
            return (action.numpy(), logp_action.numpy(), v)

    def act(self, state: Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.pi.policy(state).sample().numpy()


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
        # env states
        self.states = np.zeros([capacity, *state_shape], dtype=np.float32)
        # predicted actions
        self.actions = np.zeros([capacity, *action_shape], dtype=np.float32)
        # action advantages
        self.advantages = np.zeros(capacity, dtype=np.float32)
        # action rewards
        self.rewards = np.zeros(capacity, dtype=np.float32)
        # discounted cumulative future rewards for the state
        self.returns = np.zeros(capacity, dtype=np.float32)
        # predicted env state values
        self.values = np.zeros(capacity, dtype=np.float32)
        # action log probabilities
        self.logps = np.zeros(capacity, dtype=np.float32)
        # disctount factor
        self.gamma = gamma
        # ???
        self.lamda = lamda
        # index for the next insert
        self.next_index = 0
        # index for the start of the current episode
        self.espisode_start_index = 0
        # buffer capacity
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        logp: float,
        value: float,
        reward: float,
    ):
        self.states[self.next_index] = state
        self.actions[self.next_index] = action
        self.logps[self.next_index] = logp
        self.values[self.next_index] = value
        self.rewards[self.next_index] = reward
        self.next_index += 1

    def push_episode_end(self, value: float):
        range = slice(self.espisode_start_index, self.next_index)
        ep_values = np.append(self.values[range], np.array(value))
        ep_rewards = np.append(self.rewards[range], np.array(value))
        # calculate GAE-Lambda advantage
        # all rewards except the last one
        # plus discounted values
        # for each action, calculate R_st,at +  V_st
        deltas = ep_rewards[:-1] + self.gamma * ep_values[1:] - ep_values[:-1]  # ???
        self.advantages[range] = cumulative_sum(deltas, self.gamma * self.lamda)

        # set returns (rewards-to-go) as the cumulative sum of episode rewards
        self.returns[range] = cumulative_sum(ep_rewards, self.gamma)[:-1]
        self.espisode_start_index = self.next_index

    def get_batch(self) -> "TrajectoryBatch":
        """TODO"""
        assert self.next_index == self.capacity
        self.next_index = 0
        self.espisode_start_index = 0
        advantage_mean = np.mean(self.advantages)
        advantage_std = np.std(self.advantages)
        self.advantages = (self.advantages - advantage_mean) / advantage_std
        return TrajectoryBatch(
            states=self.states,
            actions=self.actions,
            returns=self.returns,
            advantages=self.advantages,
            logps=self.logps,
        )


@dataclass
class TrajectoryBatch:
    states: np.ndarray
    actions: np.ndarray
    advantages: np.ndarray
    logps: np.ndarray
    returns: np.ndarray


@dataclass
class PolicyInfo:
    approximate_kl: float
    mean_entropy: float
    clipped_fraction: float


def count_parameters(module: nn.Module) -> int:
    """Returns the total number of parameters in a NN module"""
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def cumulative_sum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Returns the discounted cumulative sum of vector elements
    Example: cs([1,2,3], 0.95) => [5.59325, 4.835, 3]
    """
    result = np.empty_like(x, dtype=np.float32)
    result[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        result[i] = x[i] + gamma * result[i + 1]
    return result


if __name__ == "__main__":
    agent = AntAgent()
    agent.train()
