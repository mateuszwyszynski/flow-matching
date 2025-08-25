"""
Control 1D Runner (PyGame)
-----------------------------------
A minimal image-based environment with keyboard control for up/down motion.
**Variant**:
- Static world; agent moves from left to right at constant speed.
- 2-4 static obstacle columns with vertical gaps.
- A **target column** placed after the last obstacle with its **own vertical gap**; you must cross it at the right level to finish.
- UP/DOWN apply ±delta per tick on y (clamped to [0,1]).
- Freeze-on-block: if a forward step would put the agent inside a column (obstacle or target) while outside its gap, x does not advance this tick.
- Logging: per-tick (image, y_position) plus metadata saved to .npz after each episode.

Dependencies: pygame, numpy
Run: python runner.py --seed 123
Exit: ESC to end episode early, Q to quit.
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


@dataclass
class EnvConfig:
    width: int = 250
    height: int = 90
    fc: float = 50.0
    delta_per_sec: float = 0.7            # fraction of screen per second for key-hold motion (0..1)
    agent_x_px: int = 10                  # initial x position in pixels
    agent_speed_px_s: float = 40.0        # constant forward speed (px/s)
    agent_radius_px: int = 3

    # Obstacles (static columns with vertical gap)
    num_obstacles_range: Tuple[int, int] = (2, 4)
    obstacle_width_px: int = 18
    obstacle_min_gap_px: int = 36         # must exceed 2*agent_radius_px
    obstacle_min_spacing_px: int = 24     # minimum spacing between successive obstacles
    obstacle_x_margin_left: int = 35
    obstacle_x_margin_right: int = 10

    # Target column (acts like an obstacle you must pass through at the right y)
    target_width_px: int = 6
    target_gap_px: int = 18               # gap size for target column

    # Colors
    bg_color: int = 0                     # grayscale 0..255
    agent_color: int = 200
    obstacle_color: int = 120
    target_color: int = 180

    clamp_margin_px: int = 2              # margin from top/bottom for clamping


@dataclass
class EpisodeConfig:
    seed: int = 0
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
        # Agent state
        self.y = 0.5                        # vertical position in [0,1]
        self.x_px = float(self.cfg.agent_x_px)
        # World layout
        self._generate_static_world()
        obs = self.render("rgb_array")
        return obs


    def _clamp_y_inside_gaps_given_x(self, y_px: int, ax: int, y_prev_px: int) -> int:
        """If agent's x is inside any column, clamp y to the intersection of their gaps.
        Falls back to previous y if the intersection is empty."""
        intervals = []

        # obstacles
        for x, gap_y in self.obstacles:
            left = int(x)
            right = left + self.cfg.obstacle_width_px
            if left <= ax <= right:
                low, high = self._allowed_y_interval_for_gap(gap_y, self.cfg.obstacle_min_gap_px)
                intervals.append((low, high))

        # target
        left_t = int(self.target_x_px)
        right_t = left_t + self.cfg.target_width_px
        if left_t <= ax <= right_t:
            low, high = self._allowed_y_interval_for_gap(self.target_gap_center_px, self.cfg.target_gap_px)
            intervals.append((low, high))

        # if not in any column → free vertical movement (screen clamp happens elsewhere)
        if not intervals:
            return y_px

        # intersection of all intervals
        low = max(lo for lo, hi in intervals)
        high = min(hi for lo, hi in intervals)

        if low <= high:
            # clamp into the feasible intersection
            return int(np.clip(y_px, low, high))
        else:
            # degenerate case: no intersection → keep previous y (no vertical motion)
            return y_prev_px


    def step_from_keyboard(self) -> Tuple[np.ndarray, bool]:
        """Poll keys, update agent, advance one tick. Returns (obs, reached_target)."""
        self._handle_events_nonblocking()
        keys = pygame.key.get_pressed()
        dy = 0.0
        if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            dy = +self.cfg.delta_per_sec * self.dt
        elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
            dy = -self.cfg.delta_per_sec * self.dt
        # apply vertical motion and clamp to [0,1] with pixel margin
        y_prev_px = self._y_to_px(self.y)
        
        self.y = float(np.clip(self.y + dy, 0.0, 1.0))
        y_px = self._y_to_px(self.y)
        y_px = int(np.clip(y_px, self.cfg.clamp_margin_px, self.cfg.height-1-self.cfg.clamp_margin_px))

        ax = int(round(self.x_px))
        y_px = self._clamp_y_inside_gaps_given_x(y_px, ax, y_prev_px)
        self.y = self._px_to_y(y_px)

        # Constant forward motion with **freeze-on-block** semantics
        self._advance_x_with_freeze()

        reached = self._reached_target()

        obs = self.render("rgb_array")
        self.t += 1
        return obs, reached

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        # Clear background
        self.surface.fill((self.cfg.bg_color, self.cfg.bg_color, self.cfg.bg_color))
        # Draw obstacles (static)
        for x, gap_y in self.obstacles:
            self._draw_column(x, gap_y, self.cfg.obstacle_width_px, self.cfg.obstacle_min_gap_px, self.cfg.obstacle_color)
        # Draw target column with its own gap
        self._draw_column(self.target_x_px, self.target_gap_center_px, self.cfg.target_width_px, self.cfg.target_gap_px, self.cfg.target_color)
        # Draw agent
        agent_center = (int(round(self.x_px)), self._y_to_px(self.y))
        pygame.draw.circle(self.surface, (self.cfg.agent_color,)*3, agent_center, self.cfg.agent_radius_px)

        if mode == "rgb_array":
            arr = pygame.surfarray.array3d(self.surface)  # (W,H,3)
            arr = np.transpose(arr, (1, 0, 2))            # -> (H,W,3)
            gray = (0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]).astype(np.uint8)
            return gray
        else:
            raise NotImplementedError

    def _draw_column(self, x_left: float, gap_center_px: float, width_px: int, gap_px: int, color_gray: int):
        top_rect = pygame.Rect(int(x_left), 0, width_px, max(0, int(gap_center_px - gap_px//2)))
        bot_rect = pygame.Rect(int(x_left), int(gap_center_px + gap_px//2), width_px, self.cfg.height - int(gap_center_px + gap_px//2))
        c = (color_gray,)*3
        if top_rect.height > 0:
            pygame.draw.rect(self.surface, c, top_rect)
        if bot_rect.height > 0:
            pygame.draw.rect(self.surface, c, bot_rect)

    def close(self):
        pygame.quit()

    # --------------------------- Internal mechanics --------------------------- #
    def _generate_static_world(self):
        # Choose number of obstacles
        n_min, n_max = self.cfg.num_obstacles_range
        n_obs = int(self.rng.randint(n_min, n_max+1))
        # Candidate x positions spaced across the screen
        usable_left = self.cfg.obstacle_x_margin_left
        usable_right = self.cfg.width - self.cfg.obstacle_x_margin_right

        # Evenly spread obstacle columns with small jitter; enforce minimum spacing
        x_positions: List[float] = []
        if n_obs > 0:
            base = np.linspace(usable_left, usable_right, n_obs, endpoint=False)
            jitter = self.rng.randint(-3, 3, size=n_obs)
            xs = np.clip(
                (base + jitter).astype(int),
                usable_left,
                usable_right - self.cfg.obstacle_width_px,
            )
            xs.sort()
            last = None
            for x in xs:
                min_dx = max(self.cfg.obstacle_min_spacing_px, self.cfg.obstacle_width_px + 6)
                if last is None or (x - last) >= min_dx:
                    x_positions.append(float(x))
                    last = x
                else:
                    x = int(last + min_dx)
                    x = min(x, usable_right - self.cfg.obstacle_width_px)
                    x_positions.append(float(x))
                    last = x

        # Vertical gap centers for obstacles
        gaps = [float(self.rng.randint(12, self.cfg.height - 12)) for _ in range(len(x_positions))]
        self.obstacles: List[Tuple[float, float]] = list(zip(x_positions, gaps))

        # Target at the very end of the world
        self.target_x_px = float(self.cfg.width - self.cfg.target_width_px - 3)
        self.target_gap_center_px = float(self.rng.randint(12, self.cfg.height - 12))


    def _allowed_y_interval_for_gap(self, gap_center_px: float, gap_px: int) -> Tuple[int, int]:
        r = self.cfg.agent_radius_px
        low = int(gap_center_px - gap_px // 2) + r + 1
        high = int(gap_center_px + gap_px // 2) - r - 1
        if high < low:
            low = high = int(gap_center_px)
        # clamp to screen
        low = max(low, self.cfg.clamp_margin_px)
        high = min(high, self.cfg.height - 1 - self.cfg.clamp_margin_px)
        return low, high

    def _advance_x_with_freeze(self):
        """Advance x by constant speed unless that would place the agent inside a blocking column
        (either an obstacle or the target) while outside its gap."""
        y_px = self._y_to_px(self.y)
        dx = self.cfg.agent_speed_px_s * self.dt
        tentative_x = self.x_px + dx
        ax_next = int(round(tentative_x))

        blocked = False
        # Check obstacle columns
        for x, gap_y in self.obstacles:
            left = int(x)
            right = left + self.cfg.obstacle_width_px
            if left <= ax_next <= right:
                low, high = self._allowed_y_interval_for_gap(gap_y, self.cfg.obstacle_min_gap_px)
                if not (low <= y_px <= high):
                    blocked = True
                    break
        # Check target column
        if not blocked:
            left = int(self.target_x_px)
            right = left + self.cfg.target_width_px
            if left <= ax_next <= right:
                low, high = self._allowed_y_interval_for_gap(self.target_gap_center_px, self.cfg.target_gap_px)
                if not (low <= y_px <= high):
                    blocked = True
        if not blocked:
            self.x_px = tentative_x
        # else: keep x_px; player can adjust y this tick

    def _reached_target(self) -> bool:
        # Consider target reached when front of agent passes the right edge of target column
        # while inside the target gap (or immediately after if we are already beyond and inside).
        ax = int(round(self.x_px))
        right = int(self.target_x_px) + self.cfg.target_width_px
        y_px = self._y_to_px(self.y)
        low, high = self._allowed_y_interval_for_gap(self.target_gap_center_px, self.cfg.target_gap_px)
        return (ax >= right) and (low <= y_px <= high)

    def _y_to_px(self, y: float) -> int:
        return int(round((1.0 - y) * (self.cfg.height - 1)))

    def _px_to_y(self, y_px: int) -> float:
        return 1.0 - (y_px / float(self.cfg.height - 1))

    def _handle_events_nonblocking(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
        # (no-op aside from QUIT handling)


# ----------------------------- Episode recorder ---------------------------- #
@dataclass
class EpisodeRecord:
    images: np.ndarray  # (N, H, W) uint8
    y_positions: np.ndarray  # (N,) float32 in [0,1]
    meta: Dict


def run_human_episode(env: PositionControl1DEnv, epi_cfg: EpisodeConfig) -> EpisodeRecord:
    H, W = env.cfg.height, env.cfg.width
    images = []
    ys = []

    screen = pygame.display.set_mode((env.cfg.width, env.cfg.height))
    pygame.display.set_caption("PositionControl1D Runner: UP/DOWN to move | ESC end | Q quit")

    dt_target_ms = int(round(1000.0 / env.cfg.fc))
    quit_all = False

    # Warm start render
    frame = env.render("rgb_array")
    images.append(frame)
    ys.append(env.y)

    i = 1
    reached = False
    while not reached:
        start = time.time()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    reached = True
                    break
                if event.key == pygame.K_q:
                    quit_all = True
                    reached = True
                    break

        if reached:
            break

        # Update one tick from keyboard state
        frame, reached = env.step_from_keyboard()
        images.append(frame)
        ys.append(env.y)

        # Blit to display for human feedback (optional)
        pygame.surfarray.blit_array(screen, np.transpose(np.repeat(frame[..., None], 3, axis=2), (1, 0, 2)))
        pygame.display.flip()

        # Frame pacing to ~fc Hz
        elapsed_ms = int((time.time() - start) * 1000.0)
        delay = max(0, dt_target_ms - elapsed_ms)
        pygame.time.delay(delay)
        i += 1

        if quit_all:
            break

    images = np.array(images)
    ys = np.array(ys)

    meta = {
        "seed": epi_cfg.seed,
        "seconds": float(i / env.cfg.fc),
        "fc": env.cfg.fc,
        "delta_per_sec": env.cfg.delta_per_sec,
        "width": env.cfg.width,
        "height": env.cfg.height,
        "agent_speed_px_s": env.cfg.agent_speed_px_s,
        "agent_radius_px": env.cfg.agent_radius_px,
        "num_obstacles_range": env.cfg.num_obstacles_range,
        "obstacle_width_px": env.cfg.obstacle_width_px,
        "obstacle_min_gap_px": env.cfg.obstacle_min_gap_px,
        "obstacle_min_spacing_px": env.cfg.obstacle_min_spacing_px,
        "target_x_px": getattr(env, "target_x_px", None),
        "target_gap_px": env.cfg.target_gap_px,
        "target_gap_center_px": getattr(env, "target_gap_center_px", None),
        "target_width_px": env.cfg.target_width_px,
        "agent_x_start_px": env.cfg.agent_x_px,
    }

    return EpisodeRecord(images=images, y_positions=ys, meta=meta)


def save_episode(ep: EpisodeRecord, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, images=ep.images, y_positions=ep.y_positions, meta=json.dumps(ep.meta))


# --------------------------------- CLI main -------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fc", type=float, default=50.0)
    parser.add_argument("--delta-per-sec", type=float, default=0.7)
    parser.add_argument("--save-to", type=str, default="./episodes")
    args = parser.parse_args()

    cfg = EnvConfig(fc=args.fc, delta_per_sec=args.delta_per_sec)
    epi_cfg = EpisodeConfig(seed=args.seed, save_to=args.save_to)

    env = PositionControl1DEnv(cfg, seed=epi_cfg.seed)
    env.reset(seed=epi_cfg.seed)

    try:
        rec = run_human_episode(env, epi_cfg)
    finally:
        env.close()

    # Save
    timestamp = int(time.time())
    base = os.path.join(epi_cfg.save_to, f"runner_seed{epi_cfg.seed}_fc{cfg.fc}_ts{timestamp}")
    save_episode(rec, base + ".npz")
    print(f"Saved episode to: {base}.npz")


if __name__ == "__main__":
    main()
