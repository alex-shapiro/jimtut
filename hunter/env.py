from dataclasses import dataclass
from enum import Enum
from typing import final, Never, Literal, override
import gymnasium as gym
import numpy as np
import pygame

from gymnasium.spaces import Dict, Box, Discrete
from pygame import Surface


@final
class HunterEnv(gym.Env["HunterObs", int]):
    def __init__(self, size: int = 9, render_mode: Literal["human"] | None = None):
        super().__init__()
        self.size = 9
        self.current_player = Player.PREDATOR
        self.timeout = 256  # max number of turns per game
        self.t = 0
        self.predator_pos = np.array([2, 2])
        self.prey_pos = np.array([6, 6])
        self.min_pos = np.array([0, 0])
        self.max_pos = np.array([size - 1, size - 1])
        self.render_mode = render_mode
        self.observation_space = Dict(
            {
                "current_player": Discrete(2),
                "predator": Box(0, size - 1, shape=(2,), dtype=int),
                "prey": Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = Discrete(4)
        self.action_to_direction = {
            0: np.array([0, 1]),  # up
            1: np.array([1, 0]),  # right
            2: np.array([0, -1]),  # down
            3: np.array([-1, 0]),  # left
        }

        # rendering
        self.window_size = 512
        self.window: Surface | None = None
        self.clock = None

    @override
    def reset(
        self,
        seed: int | None = None,
    ) -> tuple["HunterObs", dict[Never, Never]]:
        _ = super().reset(seed=seed)
        self.t = 0
        self.predator_pos = self.random_pos()
        self.prey_pos = self.random_pos()
        while np.equal(self.predator_pos, self.prey_pos).all():
            self.prey_pos = self.random_pos()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> "HunterObs":
        return HunterObs(
            current_player=self.current_player,
            predator_pos=self.predator_pos,
            prey_pos=self.prey_pos,
        )

    def _get_info(self) -> dict[Never, Never]:
        return {}

    def _render_frame(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            _ = pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size, self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        canvas = Surface((self.window_size, self.window_size))
        _ = canvas.fill((255, 255, 255))
        square_px = self.window_size / self.size
        _ = pygame.draw.rect(
            suface=canvas,
            color=(255, 0, 0),
            rect=pygame.Rect(square_px * self.predator_pos, (square_px, square_px)),
        )
        _ = pygame.draw.rect(
            surface=canvas,
            color=(0, 255, 0),
            rect=pygame.Rect(square_px * self.prey_pos, (square_px, square_px)),
        )
        _ = self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        _ = self.clock.tick(4)  # TODO: tune FPS

    @override
    def step(
        self, action: int
    ) -> tuple["HunterObs", float, bool, bool, dict[Never, Never]]:
        if self.t > self.timeout:
            raise RuntimeError("environment has passed its timeout")
        diff = self.action_to_direction[action]
        if self.current_player == Player.PREDATOR:
            self.predator_pos = self.add_pos(self.predator_pos, diff)
            reward_multiplier = 1
        else:
            self.prey_pos = self.add_pos(self.prey_pos, diff)
            reward_multiplier = -1

        obs = self._get_obs()
        done = bool(np.equal(self.predator_pos, self.prey_pos).all())
        base_reward = 0 if done else 1
        reward = base_reward * reward_multiplier
        truncated = self.t >= self.timeout
        info = self._get_info()
        self.t += 1
        self.current_player = self.current_player.other()
        return (obs, reward, done, truncated, info)

    def random_pos(self) -> np.ndarray:
        return self.np_random.integers(0, self.size, size=2, dtype=int)

    def add_pos(self, pos: np.ndarray, diff: np.ndarray) -> np.ndarray:
        return (pos + diff).clip(self.min_pos, self.max_pos)


@dataclass
class HunterObs:
    current_player: "Player"
    predator_pos: np.ndarray
    prey_pos: np.ndarray


class Player(Enum):
    PREDATOR = 0
    PREY = 1

    def other(self):
        return Player.PREY if self == Player.PREDATOR else Player.PREDATOR


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass
class Position:
    x: int
    y: int
