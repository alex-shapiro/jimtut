from dataclasses import dataclass
from typing import final

import gymnasium as gym
import numpy as np

from chart import plot_rewards


SEED = 58922320


@dataclass
class TrainingResults:
    reward_mean: float
    reward_std: float
    rewards: list[float]


@final
class BlackjackAgent:
    def __init__(
        self,
        eps_initial: float = 1.0,
        eps_final: float = 0.1,
        lr: float = 0.01,
        discount_factor: float = 0.95,
    ):
        # Blackjack sim environment
        self.env = gym.make("Blackjack-v1")
        # Epsilon (probability of a random choice)
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps = self.eps_initial
        # Learning rate
        self.lr = lr
        # Future reward discount factor
        self.discount_factor = discount_factor
        # Q learning array
        (player_sum, dealer_value, usable_ace) = self.env.observation_space
        self.n_actions = self.env.action_space.n
        self.q = np.zeros([player_sum.n, dealer_value.n, usable_ace.n, self.n_actions])

    def eval(self, seed: int = SEED) -> float:
        """Eval a single game of blackjack using the greedy strategy"""
        (player_sum, dealer_value, usable_ace), _ = self.env.reset(seed=seed)
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            action = np.argmax(self.q[player_sum, dealer_value, usable_ace])
            (player_sum, dealer_value, usable_ace), reward, done, truncated, _ = (
                self.env.step(action)
            )
            total_reward += float(reward)
        return total_reward

    def train(self, n_episodes: int = 10_000, seed: int = SEED) -> TrainingResults:
        """Train a batch of episodes"""
        rewards: list[float] = []
        eps = self.eps_initial
        eps_decay_rate = (eps - self.eps_final) / n_episodes
        for i in range(n_episodes):
            reward = self.__train_episode(eps=eps, seed=seed + i)
            rewards.append(reward)
            eps -= eps_decay_rate

        return TrainingResults(
            reward_mean=float(np.mean(rewards)),
            reward_std=float(np.std(rewards)),
            rewards=rewards,
        )

    def __train_episode(self, eps: float, seed: int):
        (player_sum, dealer_value, usable_ace), _ = self.env.reset(seed=seed)
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            # choose an action
            action = 0
            if np.random.random() < eps:
                action = int(self.env.action_space.sample())
            else:
                action = np.argmax(self.q[player_sum, dealer_value, usable_ace])

            # step the environment
            (
                (next_player_sum, next_dealer_value, next_usable_ace),
                reward,
                done,
                truncated,
                _,
            ) = self.env.step(action)
            reward = float(reward)  # for type checks only
            total_reward += reward
            # update Q reward array
            q_now = float(self.q[player_sum, dealer_value, usable_ace, action])
            if done or truncated:
                td_err = reward - q_now
                self.q[player_sum, dealer_value, usable_ace, action] += self.lr * td_err
            else:
                q_next = np.max(
                    self.q[next_player_sum, next_dealer_value, next_usable_ace]
                )
                td_err = reward + self.discount_factor * q_next - q_now
                self.q[player_sum, dealer_value, usable_ace, action] += self.lr * td_err

            player_sum = next_player_sum
            dealer_value = next_dealer_value
            usable_ace = next_usable_ace
        return total_reward


def main():
    seed = SEED
    agent = BlackjackAgent()
    train_results = agent.train(n_episodes=100_000, seed=seed)
    eval_results = [agent.eval(seed=i) for i in range(1000)]
    print(f"mean: {np.mean(eval_results)}")
    print(f"std: {np.std(eval_results)}")
    plot_rewards(
        train_results.rewards, title="Blackjack Training Rewards", batch_size=100
    )


if __name__ == "__main__":
    main()
