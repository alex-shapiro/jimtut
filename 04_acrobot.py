from typing import final, override

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete
from torch import Tensor, nn
from torch.optim import RMSprop

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
        self.env.action_space.seed(seed)  # pyright: ignore[reportUnusedCallResult]
        self.seed = seed

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=100_000,
            state_space=state_space,
            action_space=action_space,
        )

        # exploration hyperparameters
        self.n_steps = 100_000
        self.eps_initial = 1.0
        self.eps_final = 0.04
        self.exploration_fraction = 0.1

        # other hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.batch_size = 128

        # models
        state_size = state_space.shape[0]
        action_size = int(action_space.n)
        self.device = torch.device("cpu")
        self.q_net = QNet(state_size, action_size).to(self.device)
        self.target_net = QNet(state_size, action_size).to(self.device)
        self.optimizer = RMSprop(self.q_net.parameters(), lr=self.learning_rate)

    def train(
        self,
        learning_interval: int = 2,
        target_update_interval: int = 10,
        eval_interval: int = 5000,
        n_eval_episodes: int = 100,
    ):
        state, _ = self.env.reset(seed=self.seed)
        for step in range(1, self.n_steps + 1):
            eps = self.exploration_rate(step)
            state = self.step(state, eps)

            if step % learning_interval == 0:
                self.teach_qnet()

            if step % target_update_interval == 0:
                _ = self.target_net.load_state_dict(self.q_net.state_dict())

            if step % eval_interval == 0:
                rewards = self.evaluate(n_episodes=n_eval_episodes)
                r_mean = np.mean(rewards)
                r_std = np.std(rewards)
                print()
                print(f"Evaluation at step {step} (eps = {eps:.2f})")
                print(f"Reward: {r_mean:.2f} +/- {r_std:.2f}")

    def step(self, state: np.ndarray, eps: float) -> np.ndarray:
        """Runs the next simulator step and returns the resulting state"""
        action = self.epsilon_greedy_action(state, eps, self.q_net)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        self.replay_buffer.push(
            state=state,
            next_state=next_state,
            action=action,
            reward=float(reward),
            terminated=(terminated or truncated),
        )
        if terminated or truncated:
            next_state, _ = self.env.reset()
        return next_state

    def teach_qnet(self):
        """Updates the agent's QNet from a ReplayBuffer batch"""
        batch = self.replay_buffer.sample(self.batch_size).to_torch(self.device)

        # get current action Q value
        current_q_value: Tensor = self.q_net(batch.states)  # pyright: ignore[reportAny]
        current_q_value = current_q_value.gather(1, batch.actions)
        current_q_value = current_q_value.squeeze(dim=-1)

        # get TD target from the next_state's max Q value
        with torch.no_grad():
            next_q_value: Tensor = self.target_net(batch.next_states)  # pyright: ignore[reportAny]
            next_q_value, _ = next_q_value.max(dim=1)
            should_boostrap = batch.terminateds.logical_not()
            td_target = (
                batch.rewards + self.discount_factor * next_q_value * should_boostrap
            )

        # sanity check
        assert current_q_value.shape == (self.batch_size,)
        assert current_q_value.shape == td_target.shape

        # compute MSE loss and backprop
        loss = F.mse_loss(current_q_value, td_target)
        self.optimizer.zero_grad()
        loss.backward()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]

    def evaluate(self, n_episodes: int, eps: float = 0.0):
        """Evaluates the agent"""
        episode_rewards: list[float] = []
        for _ in range(n_episodes):
            total_reward = 0.0
            state, _ = self.eval_env.reset(seed=self.seed)
            done = False
            while not done:
                action = self.epsilon_greedy_action(state, eps, self.q_net)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            episode_rewards.append(total_reward)
        return episode_rewards

    def epsilon_greedy_action(
        self, state: np.ndarray, eps: float, q_net: "QNet"
    ) -> int:
        if np.random.rand() < eps:
            return int(self.env.action_space.sample())
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
            q_values: Tensor = q_net(state_tensor)  # pyright: ignore[reportAny]
            return int(q_values.argmax().item())

    def exploration_rate(self, step: int) -> float:
        """Returns the current exploration rate according to a linearly decreasing schedule"""
        explore_steps = self.exploration_fraction * self.n_steps
        progress = min(1.0, step / explore_steps)
        return self.eps_initial + progress * (self.eps_final - self.eps_initial)


@final
class QNet(nn.Module):
    """NN for Q-value prediction"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(QNet, self).__init__()  # pyright: ignore[reportUnknownMemberType]
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    @override
    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # pyright: ignore[reportAny]


if __name__ == "__main__":
    agent = AcrobotAgent()
    agent.train()
