"""Utility for plotting reward charts"""

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(
    rewards: list[float],
    title: str = "Rewards Over Time",
    batch_size: int = 10,
    smoothing_window: int = 100,
) -> None:
    """Plot a chart of rewards over time, averaged over batches with smoothing

    Args:
        rewards: List of reward values over time
        title: Chart title (default: "Rewards Over Time")
        batch_size: Number of episodes to average together (default: 10)
        smoothing_window: Window size for moving average smoothing (default: 100)
    """
    # Calculate mean rewards for each batch
    n_batches = len(rewards) // batch_size
    batched_rewards = []
    for i in range(n_batches):
        batch = rewards[i * batch_size : (i + 1) * batch_size]
        batched_rewards.append(np.mean(batch))

    # X-axis: middle episode number of each batch
    x_values = [(i * batch_size + batch_size // 2) for i in range(n_batches)]

    # Calculate smoothed trendline using moving average
    smoothed = []
    for i in range(len(batched_rewards)):
        start_idx = max(0, i - smoothing_window // 2)
        end_idx = min(len(batched_rewards), i + smoothing_window // 2)
        window = batched_rewards[start_idx:end_idx]
        smoothed.append(np.mean(window))

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, batched_rewards, linewidth=0.5, alpha=0.3, label="Batched")
    plt.plot(
        x_values,
        smoothed,
        linewidth=2,
        alpha=0.9,
        label=f"Smoothed (window={smoothing_window})",
    )
    plt.xlabel(f"Episode (batched by {batch_size})")
    plt.ylabel("Mean Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()
