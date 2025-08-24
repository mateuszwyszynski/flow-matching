"""
Position-Control 1D Scroller (PyGame)
-------------------------------------
A minimal image-based environment with keyboard control for up/down motion.
- Observation: single grayscale frame (H x W) with agent on the left and right-to-left scrolling obstacles.
- Action (during human play): holding UP/DOWN applies a fixed per-tick position increment ±delta (clamped to [0,1]).
- Logging: per-tick (image, y_position) plus metadata saved to .npz after each episode.
- NEW: Solid collision model with **world-freeze**. If an obstacle column overlaps the agent's x
  and the agent's y is outside the vertical gap (considering radius), the world **does not scroll** this tick.
  This prevents "teleporting" the dot and lets the player reposition into the gap.

Dependencies: pygame, numpy
Run: python position_control_1d_env.py --seed 123 --ep-seconds 20 --fc 50 --delta-per-sec 0.7 --save-to ./episodes
Exit: ESC to end episode early, Q to quit entirely.
"""
from __future__ import annotations
import os
import time
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import pygame
except Exception:
    raise SystemExit("Please install pygame: pip install pygame")


# ----------------------------- Config dataclass ----------------------------- #
@dataclass
class EnvConfig:
    width: int = 240
    height: int = 84
    fc: float = 50.0                      # control frequency (Hz)
    pixels_per_second: float = 60.0       # obstacle scroll speed (px/s)
    delta_per_sec: float = 0.7            # fraction of screen per second for key-hold motion (0..1)
    agent_x_px: int = 12                  # agent x location in pixels
    agent_radius_px: int = 3
    obstacle_min_gap_px: int = 22         # vertical gap between bars (must exceed 2*agent_radius_px)
    obstacle_width_px: int = 18
    obstacle_spawn_interval_s: Tuple[float, float] = (1.3, 1.6)  # randomized spawn interval
    bg_color: int = 0                     # grayscale 0..255
    agent_color: int = 200
    obstacle_color: int = 120
    clamp_margin_px: int = 2              # margin from top/bottom for clamping


@dataclass
class EpisodeConfig:
    seed: int = 0
    seconds: float = 20.0
    save_to: str = "./episodes"


# ----------------------------- Environment core ---------------------------- #
class PositionControl1DEnv:
    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        pygame.init()
        self.surface = pygame.Surface((cfg.width, cfg.height))
        self.clock = pygame.time.Clock()
        self.reset(seed)

    # Public API
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.t = 0
        self.dt = 1.0 / self.cfg.fc
        # Agent vertical position in [0,1]
        self.y = 0.5
        self._init_obstacles()
        obs = self.render("rgb_array")
        return obs

    def step_from_keyboard(self) -> np.ndarray:
        """Poll keys, update agent, advance world by one tick, return obs.
        World-freeze: if agent cannot legally pass through the obstacle column at its x,
        we skip world scrolling and obstacle spawning for this tick (player can adjust y)."""
        self._handle_events_nonblocking()
        keys = pygame.key.get_pressed()
        dy = 0.0
        if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            dy = +self.cfg.delta_per_sec * self.dt
        elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
            dy = -self.cfg.delta_per_sec * self.dt
        # both or none -> dy = 0
        self.y = float(np.clip(self.y + dy, 0.0, 1.0))

        # clamp with small pixel margin so the dot remains visible
        y_px = self._y_to_px(self.y)
        y_px = int(np.clip(y_px, self.cfg.clamp_margin_px, self.cfg.height-1-self.cfg.clamp_margin_px))
        self.y = self._px_to_y(y_px)

        # Advance the world with freeze semantics
        self._advance_world_with_freeze()

        # Render
        obs = self.render("rgb_array")
        self.t += 1
        return obs

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        # Clear background
        self.surface.fill((self.cfg.bg_color, self.cfg.bg_color, self.cfg.bg_color))
        # Draw obstacles
        for x, gap_y in self.obstacles:
            gap = self.cfg.obstacle_min_gap_px
            top_rect = pygame.Rect(int(x), 0, self.cfg.obstacle_width_px, max(0, int(gap_y - gap//2)))
            bot_rect = pygame.Rect(int(x), int(gap_y + gap//2), self.cfg.obstacle_width_px, self.cfg.height - int(gap_y + gap//2))
            c = (self.cfg.obstacle_color,)*3
            if top_rect.height > 0:
                pygame.draw.rect(self.surface, c, top_rect)
            if bot_rect.height > 0:
                pygame.draw.rect(self.surface, c, bot_rect)
        # Draw agent
        agent_center = (self.cfg.agent_x_px, self._y_to_px(self.y))
        pygame.draw.circle(self.surface, (self.cfg.agent_color,)*3, agent_center, self.cfg.agent_radius_px)

        if mode == "rgb_array":
            arr = pygame.surfarray.array3d(self.surface)  # (W,H,3)
            arr = np.transpose(arr, (1, 0, 2))            # -> (H,W,3)
            # Grayscale conversion (luminosity method)
            gray = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]).astype(np.uint8)
            return gray
        else:
            raise NotImplementedError

    def close(self):
        pygame.quit()

    # --------------------------- Internal mechanics --------------------------- #
    def _init_obstacles(self):
        self.obstacles: List[Tuple[float, float]] = []  # list of (x_px, gap_center_y_px)
        self.next_spawn_in = self._sample_spawn_interval()

    def _sample_spawn_interval(self) -> float:
        a, b = self.cfg.obstacle_spawn_interval_s
        return float(self.rng.uniform(a, b))

    def _y_to_px(self, y: float) -> int:
        # y in [0,1] -> pixel row (0 top)
        return int(round((1.0 - y) * (self.cfg.height - 1)))

    def _px_to_y(self, y_px: int) -> float:
        return 1.0 - (y_px / float(self.cfg.height - 1))

    def _handle_events_nonblocking(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
        # (no-op aside from QUIT handling)

    # ---- Collision helpers (pixel space) ---- #
    def _allowed_y_interval_for_column(self, gap_center_px: float) -> Tuple[int, int]:
        gap = self.cfg.obstacle_min_gap_px
        r = self.cfg.agent_radius_px
        low = int(gap_center_px - gap // 2) + r + 1
        high = int(gap_center_px + gap // 2) - r - 1
        if high < low:
            low = high = int(gap_center_px)
        # clamp to screen
        low = max(low, self.cfg.clamp_margin_px)
        high = min(high, self.cfg.height - 1 - self.cfg.clamp_margin_px)
        return low, high

    def _blocked_at_agent_x(self, obstacles: List[Tuple[float, float]], y_px: int) -> bool:
        ax = self.cfg.agent_x_px
        for x, gap_y in obstacles:
            left = int(x)
            right = left + self.cfg.obstacle_width_px
            if left <= ax <= right:
                low, high = self._allowed_y_interval_for_column(gap_y)
                if not (low <= y_px <= high):
                    return True
        return False

    def _advance_world_with_freeze(self):
        """Try to advance obstacles and spawn timers; if that would place a column on the agent
        while agent is outside the allowed gap, **do not** move the world this tick."""
        dt = self.dt
        dx = -self.cfg.pixels_per_second * dt

        # Tentative move/spawn
        tentative_obstacles = [(x + dx, y) for (x, y) in self.obstacles if (x + self.cfg.obstacle_width_px + dx) > 0]
        tentative_next_spawn = self.next_spawn_in - dt
        if tentative_next_spawn <= 0:
            x0 = float(self.cfg.width)
            gap_center = float(self.rng.randint(12, self.cfg.height - 12))
            tentative_obstacles.append((x0, gap_center))
            # reset spawn timer by sampling the next interval
            tentative_next_spawn = self._sample_spawn_interval()

        # Check if this tentative world state would block the agent at its current y
        y_px = self._y_to_px(self.y)
        if self._blocked_at_agent_x(tentative_obstacles, y_px):
            # Freeze: keep current obstacles and timers unchanged
            return
        else:
            # Commit tentative world
            self.obstacles = tentative_obstacles
            self.next_spawn_in = tentative_next_spawn


# ----------------------------- Episode recorder ---------------------------- #
@dataclass
class EpisodeRecord:
    images: np.ndarray  # (N, H, W) uint8
    y_positions: np.ndarray  # (N,) float32 in [0,1]
    meta: Dict


def run_human_episode(env: PositionControl1DEnv, epi_cfg: EpisodeConfig) -> EpisodeRecord:
    H, W = env.cfg.height, env.cfg.width
    N = int(round(epi_cfg.seconds * env.cfg.fc))
    images = np.empty((N, H, W), dtype=np.uint8)
    ys = np.empty((N,), dtype=np.float32)

    screen = pygame.display.set_mode((env.cfg.width, env.cfg.height))
    pygame.display.set_caption("PositionControl1D: UP/DOWN to move | ESC end episode | Q quit")

    dt_target_ms = int(round(1000.0 / env.cfg.fc))
    quit_all = False

    # Warm start render
    frame = env.render("rgb_array")
    images[0] = frame
    ys[0] = env.y

    i = 1
    while i < N:
        start = time.time()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # end episode early
                    N = i
                    images = images[:N]
                    ys = ys[:N]
                    break
                if event.key == pygame.K_q:
                    quit_all = True
                    N = i
                    images = images[:N]
                    ys = ys[:N]
                    break

        # Update one tick from keyboard state
        frame = env.step_from_keyboard()
        images[i] = frame
        ys[i] = env.y

        # Blit to display for human feedback (optional but helpful)
        pygame.surfarray.blit_array(screen, np.transpose(np.repeat(frame[..., None], 3, axis=2), (1, 0, 2)))
        pygame.display.flip()

        # Frame pacing to ~fc Hz
        elapsed_ms = int((time.time() - start) * 1000.0)
        delay = max(0, dt_target_ms - elapsed_ms)
        pygame.time.delay(delay)
        i += 1

        if quit_all:
            break

    meta = {
        "seed": epi_cfg.seed,
        "seconds": float(i / env.cfg.fc),
        "fc": env.cfg.fc,
        "delta_per_sec": env.cfg.delta_per_sec,
        "pixels_per_second": env.cfg.pixels_per_second,
        "width": env.cfg.width,
        "height": env.cfg.height,
        "obstacle_min_gap_px": env.cfg.obstacle_min_gap_px,
        "obstacle_width_px": env.cfg.obstacle_width_px,
        "spawn_interval_s": env.cfg.obstacle_spawn_interval_s,
        "agent_x_px": env.cfg.agent_x_px,
        "agent_radius_px": env.cfg.agent_radius_px,
    }

    return EpisodeRecord(images=images, y_positions=ys, meta=meta)


def save_episode(ep: EpisodeRecord, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, images=ep.images, y_positions=ep.y_positions, meta=json.dumps(ep.meta))


# --------------------------------- CLI main -------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ep-seconds", type=float, default=20.0)
    parser.add_argument("--fc", type=float, default=50.0)
    parser.add_argument("--delta-per-sec", type=float, default=0.7)
    parser.add_argument("--scroll-pps", type=float, default=60.0, help="obstacle scroll speed (pixels per second)")
    parser.add_argument("--save-to", type=str, default="./episodes")
    args = parser.parse_args()

    cfg = EnvConfig(fc=args.fc, delta_per_sec=args.delta_per_sec, pixels_per_second=args.scroll_pps)
    epi_cfg = EpisodeConfig(seed=args.seed, seconds=args.ep_seconds, save_to=args.save_to)

    env = PositionControl1DEnv(cfg, seed=epi_cfg.seed)
    env.reset(seed=epi_cfg.seed)

    try:
        rec = run_human_episode(env, epi_cfg)
    finally:
        env.close()

    # Save
    timestamp = int(time.time())
    base = os.path.join(epi_cfg.save_to, f"episode_seed{epi_cfg.seed}_fc{cfg.fc}_ts{timestamp}")
    save_episode(rec, base + ".npz")
    print(f"Saved episode to: {base}.npz")


if __name__ == "__main__":
    main()
