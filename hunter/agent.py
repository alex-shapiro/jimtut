import numpy as np

from typing import final
from hunter.env import HunterEnv, HunterObs, Player


@final
class HunterGame:
    def __init__(self, seed: int = 2025, epochs: int = 50):
        self.env = HunterEnv()
        self.eval_env = HunterEnv(render_mode="human")
        self.predator = PlayerAgent()
        self.prey = PlayerAgent()

    def evaluate(self, n_episodes: int = 1):
        for i in range(n_episodes):
            obs, _ = self.eval_env.reset()
            ended = False
            predator_reward = 0.0
            prey_reward = 0.0

            while not ended:
                if obs.current_player == Player.PREDATOR:
                    action = self.predator.act(obs)
                    obs, reward, done, terminated = self.eval_env.step(action)
                    predator_reward += reward
                else:
                    action = self.prey.act(obs)
                    obs, reward, done, terminated = self.eval_env.step(action)
                    prey_reward += reward
                ended = done or terminated

            print(f"EVAL (PREDATOR: {predator_reward:.2f}, PREY: {prey_reward:.2f})")


@final
class PlayerAgent:
    def __init__(self):
        self.policy_net: int = 0
        self.value_net: int = 0

    def select_action(self, state: HunterObs) -> int:
        return np.random.randint(0, 4)


if __name__ == "__main__":
    game = HunterGame()
    game.evaluate()
