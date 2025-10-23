# Acrobot, this time with simple Policy Optimization (not PPO)
# It doesn't do much better than DQN but it is slightly more consistent at hill-climbing.

from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from typing import final, override
from torch import Tensor, float32, int32, nn
from torch.distributions import Categorical
from torch.optim import AdamW


@final
class AcrobotPOAgent:
    def __init__(self, seed: int = 42):
        # sim environment
        self.env: gym.Env[np.ndarray, int] = gym.make("Acrobot-v1")  # pyright: ignore[reportUnknownMemberType]
        state_space: Box = self.env.observation_space  # pyright: ignore[reportAssignmentType]
        action_space: Discrete = self.env.action_space  # pyright: ignore[reportAssignmentType]

        # RNG seed
        np.random.seed(seed)
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.env.action_space.seed(seed)  # pyright: ignore[reportUnusedCallResult]
        self.seed = seed

        # training & model
        self.device = torch.device("cpu")
        self.policy_model = PolicyModel(
            d_state=state_space.shape[0],
            d_action=action_space.n,
        )
        self.lr = 1e-2
        self.optimizer = AdamW(self.policy_model.parameters(), lr=self.lr)

    def train(
        self,
        n_epochs: int = 20,
        n_steps_per_epoch: int = 5000,
        n_eval_episodes: int = 10,
    ):
        state, _ = self.env.reset(seed=self.seed)
        for epoch in range(n_epochs):
            print()
            print(f"epoch {epoch}")
            with torch.no_grad():
                batch = self.simulate_epoch(n_steps_per_epoch)
            self.optimizer.zero_grad()
            batch_loss = self.compute_loss(batch)
            print(f"loss: {batch_loss:.2f}")
            batch_loss.backward()
            self.optimizer.step()
            self.evaluate(n_episodes=n_eval_episodes)
            self.env.reset()

    def simulate_epoch(self, n_steps: int) -> "Batch":
        step = 0
        batch = Batch(states=[], actions=[], rewards=[])
        while step < n_steps:
            state, _ = self.env.reset()
            episode_reward = 0
            episode_len = 0
            done = False
            while not done:
                batch.states.append(state)
                action = self.get_action(torch.as_tensor(state, dtype=float32))
                batch.actions.append(action)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += float(reward)
                episode_len += 1
            batch.rewards.extend([episode_reward] * episode_len)
            step += episode_len
        print("simulated epoch")
        return batch

    def get_action(self, state: Tensor) -> int:
        action_logits: Tensor = self.policy_model(state)
        return Categorical(logits=action_logits).sample().item()

    def compute_loss(self, batch: "Batch") -> Tensor:
        batch = batch.to_torch()
        action_logits = self.policy_model(batch.states)
        action_probs = Categorical(logits=action_logits)
        action_logprobs = action_probs.log_prob(batch.actions)
        normalized_rewards = (batch.rewards - batch.rewards.mean()) / (
            batch.rewards.std() + 1e-8
        )
        return -(action_logprobs * normalized_rewards).mean()

    def evaluate(self, n_episodes: int, visible: bool = False):
        env = gym.make("Acrobot-v1", render_mode="human") if visible else self.env  # pyright: ignore[reportUnknownMemberType]
        episode_rewards: list[float] = []
        for _ in range(n_episodes):
            total_reward = 0.0
            state, _ = env.reset()
            done = False
            while not done:
                action = self.get_action(torch.as_tensor(state))
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            episode_rewards.append(total_reward)
        r_mean = np.mean(episode_rewards)
        r_std = np.std(episode_rewards)
        print(f"mean: {r_mean:.2f} +/- {r_std:.2f}")


class PolicyModel(nn.Module):
    def __init__(self, d_state: int, d_action: int, d_hidden: int = 32):
        super(PolicyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_state, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_action),
        )

    @override
    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


@dataclass
class TorchBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor


@dataclass
class Batch:
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]

    def to_torch(self) -> TorchBatch:
        return TorchBatch(
            states=torch.as_tensor(np.array(self.states), dtype=float32),
            actions=torch.as_tensor(np.array(self.actions), dtype=int32),
            rewards=torch.as_tensor(np.array(self.rewards), dtype=float32),
        )


if __name__ == "__main__":
    agent = AcrobotPOAgent()
    agent.train()
    agent.evaluate(n_episodes=3, visible=True)
