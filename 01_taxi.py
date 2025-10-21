import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from numpy._typing import NDArray

SEED = 58922320


@dataclass
class AgentResults:
    episode_rewards: list[float]
    mean_reward: float
    std_reward: float


def train_agent(
    env: gym.Env[int, int],
    episodes: int = 5_000,
    seed: int = SEED,
    lr: float = 0.1,
    discount_factor: float = 0.97,
    epsilon: float = 0.1,
) -> AgentResults:
    """Train a Q-learning agent"""
    np.random.seed(seed)
    random.seed(seed)

    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    q_table = np.zeros(shape=(n_states, n_actions), dtype=np.float64)

    print(f"n_states: {n_states}, n_actions: {n_actions}")

    episode_rewards: list[float] = []

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        done = False
        truncated = False

        # Policy
        while not (done or truncated):
            action_mask: NDArray[np.int8] = info["action_mask"]
            valid_actions = np.nonzero(action_mask == 1)[0]
            action = 0
            if np.random.random() < epsilon:
                action = int(np.random.choice(valid_actions))
            else:
                action = int(valid_actions[np.argmax(q_table[state, valid_actions])])
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)

            # Q-learning update
            if done or truncated:
                next_max = 0
            else:
                next_mask: NDArray[np.int8] = info["action_mask"]
                valid_next_actions = np.nonzero(next_mask == 1)[0]
                next_max: float = 0.0
                if len(valid_next_actions) > 0:
                    next_max = np.max(q_table[next_state, valid_next_actions])

            q_table[state, action] += lr * (
                float(reward)
                + (float(discount_factor) * next_max)
                - q_table[state, action]
            )

            state = next_state

        episode_rewards.append(total_reward)

    return AgentResults(
        episode_rewards=episode_rewards,
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
    )


def main():
    n_runs = 1
    seeds = [SEED + i for i in range(n_runs)]
    results_list: list[AgentResults] = []

    for i, seed in enumerate(seeds):
        print(f"Run {i + 1}/{n_runs} with seed {seed}")
        env: gym.Env[int, int] = gym.make("Taxi-v3")
        results = train_agent(env=env)
        env.close()
        results_list.append(results)

    mean_rewards = [r.mean_reward for r in results_list]
    overall_mean = np.mean(mean_rewards)
    overall_std = np.std(mean_rewards)

    # Create visualization
    plt.figure(figsize=(12, 8), dpi=100)

    # Plot individual runs with low alpha
    for i, results in enumerate(results_list):
        plt.plot(
            results.episode_rewards,
            label="With Action Masking" if i == 0 else None,
            color="blue",
            alpha=0.1,
        )

    # Calculate and plot mean curves across all runs
    masked_mean_curve = np.mean([r.episode_rewards for r in results_list], axis=0)
    print(masked_mean_curve)

    plt.plot(
        masked_mean_curve, label="With Action Masking (Mean)", color="blue", linewidth=2
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance: Q-Learning with Action Masking")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure
    savefig_folder = Path("_static/img/tutorials/")
    savefig_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        savefig_folder / "taxi_v3_action_masking_comparison.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.show()


if __name__ == "__main__":
    main()
