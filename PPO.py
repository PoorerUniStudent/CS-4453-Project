"""
PPO for Gymnasium CarRacing-v3 (single-file, runnable)
======================================================

This script implements Proximal Policy Optimization (PPO) for CarRacing-v3
using a discrete action space and includes command-line run settings for:

- training from scratch
- resuming training from checkpoints
- evaluation / rendering
- CSV logging every N steps
- checkpoint saving on a fixed interval

Important:
This version only keeps CLI arguments that also exist in the SAC.py file you sent,
and only uses the ones that are actually needed for this PPO script. The PPO
learning logic itself is unchanged unless necessary for checkpoint/logging support.

Reference workflow adapted from your SAC.py file: :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ============================================================
# Default Hyperparameters
# ============================================================

TOTAL_TIMESTEPS = 1_000_000
ROLLOUT_STEPS = 2048
EPOCHS = 10
MINIBATCH_SIZE = 256
LEARNING_RATE = 2.5e-4
ADAM_EPS = 1e-5
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03

FRAME_STACK = 4
WIDTH = 84
HEIGHT = 84
CROP_TOP = 0
CROP_BOTTOM = 12
REWARD_CLIP = 1.0
GRASS_TIMEOUT = 100

SEED = 42
EVAL_EVERY = 10_000
NUM_EVAL_EPS = 5


# ============================================================
# Device Selection
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[PPO] Using device: {DEVICE}")


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def maybe_init_csv(csv_path: str) -> None:
    """Create CSV file with header if it does not already exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "eval_reward",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clipfrac",
                "lr",
            ])


def append_csv_row(csv_path: str, row: List[object]) -> None:
    """Append one row to the CSV log."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================
# Discrete action wrapper
# ============================================================

class DiscreteCarRacingWrapper(gym.ActionWrapper):
    """
    Converts CarRacing continuous controls [steer, gas, brake]
    into a discrete action space.

    Action mapping:
        0: do nothing
        1: steer left
        2: steer right
        3: gas
        4: brake
        5: gas + left
        6: gas + right
        7: brake + left
        8: brake + right
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(9)

        # v1:
            # 0: np.array([0.0, 0.0, 0.0], dtype=np.float32),
            # 1: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            # 2: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            # 3: np.array([0.0, 0.6, 0.0], dtype=np.float32),
            # 4: np.array([0.0, 0.0, 0.9], dtype=np.float32),
            # 5: np.array([-0.6, 0.4, 0.0], dtype=np.float32),
            # 6: np.array([0.6, 0.4, 0.0], dtype=np.float32),
            # 7: np.array([-0.6, 0.0, 0.9], dtype=np.float32),
            # 8: np.array([0.6, 0.0, 0.9], dtype=np.float32),
        
        self._actions = {
            0: np.array([0.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            3: np.array([0.0, 0.4, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 0.9], dtype=np.float32),
            5: np.array([-0.6, 0.25, 0.0], dtype=np.float32),
            6: np.array([0.6, 0.25, 0.0], dtype=np.float32),
            7: np.array([-0.6, 0.0, 0.9], dtype=np.float32),
            8: np.array([0.6, 0.0, 0.9], dtype=np.float32),
        }

    def action(self, action: int) -> np.ndarray:
        """Map discrete action index to continuous control vector."""
        return self._actions[int(action)].copy()


# ============================================================
# Observation preprocessing
# ============================================================

class CarRacingPreprocess(gym.Wrapper):
    """
    Preprocess observations for CarRacing:
    - crop bottom HUD
    - grayscale
    - resize
    - stack consecutive frames
    - optional reward clipping
    - timeout if the car stays too long off-track
    """

    def __init__(
        self,
        env: gym.Env,
        frame_stack: int = FRAME_STACK,
        width: int = WIDTH,
        height: int = HEIGHT,
        crop_top: int = CROP_TOP,
        crop_bottom: int = CROP_BOTTOM,
        reward_clip: float | None = REWARD_CLIP,
        grass_timeout: int = GRASS_TIMEOUT,
    ):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.width = width
        self.height = height
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.reward_clip = reward_clip
        self.grass_timeout = grass_timeout

        self.frames: Deque[np.ndarray] = collections.deque(maxlen=frame_stack)
        self.offtrack_counter = 0

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(frame_stack, height, width),
            dtype=np.float32,
        )

    def _process(self, obs: np.ndarray) -> np.ndarray:
        """Crop, grayscale, resize, and normalize a single RGB frame."""
        if self.crop_bottom > 0:
            obs = obs[self.crop_top: obs.shape[0] - self.crop_bottom, :, :]
        else:
            obs = obs[self.crop_top:, :, :]

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        processed = resized.astype(np.float32) / 255.0
        return processed

    @staticmethod
    def _estimate_offtrack(frame: np.ndarray) -> bool:
        """Lightweight off-track heuristic based on center patch brightness."""
        h, w = frame.shape
        center = frame[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]
        mean_val = float(center.mean())
        return mean_val < 0.35

    def _get_stacked_obs(self) -> np.ndarray:
        """Return the current stacked observation."""
        while len(self.frames) < self.frame_stack:
            self.frames.append(self.frames[-1].copy())
        return np.stack(list(self.frames), axis=0).astype(np.float32)

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        proc = self._process(obs)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(proc)
        self.offtrack_counter = 0
        return self._get_stacked_obs(), info

    def step(self, action):
        """Step environment, preprocess, clip reward, and apply grass timeout."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        proc = self._process(obs)
        self.frames.append(proc)

        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        if self._estimate_offtrack(proc):
            self.offtrack_counter += 1
        else:
            self.offtrack_counter = 0

        if self.offtrack_counter >= self.grass_timeout:
            truncated = True
            info = dict(info)
            info["grass_timeout"] = True

        return self._get_stacked_obs(), reward, terminated, truncated, info


# ============================================================
# PPO rollout memory
# ============================================================

class PPOMemory:
    """Stores one on-policy rollout for PPO."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ):
        """Add one transition to the rollout."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.states)


# ============================================================
# Actor-Critic network
# ============================================================

class ActorCritic(nn.Module):
    """CNN-based actor-critic for stacked grayscale frames."""

    def __init__(self, input_channels: int, num_actions: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            feat_dim = self.features(dummy).shape[1]

        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and state value."""
        x = self.features(x)
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action (or score a provided one), and return:
        action, log_prob, entropy, value
        """
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


@dataclass
class PPOConfig:
    total_timesteps: int = TOTAL_TIMESTEPS
    rollout_steps: int = ROLLOUT_STEPS
    epochs: int = EPOCHS
    minibatch_size: int = MINIBATCH_SIZE
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    clip_coef: float = CLIP_COEF
    ent_coef: float = ENT_COEF
    vf_coef: float = VF_COEF
    max_grad_norm: float = MAX_GRAD_NORM
    learning_rate: float = LEARNING_RATE
    adam_eps: float = ADAM_EPS
    anneal_lr: bool = True
    target_kl: float | None = TARGET_KL


class PPOAgent:
    """PPO agent wrapper around the ActorCritic network and optimizer."""

    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int, cfg: PPOConfig, device: torch.device):
        c, _, _ = obs_shape
        self.cfg = cfg
        self.device = device
        self.net = ActorCritic(c, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select one action for a single state."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, value = self.net.get_action_and_value(state_t)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def value(self, state: np.ndarray) -> float:
        """Compute value for a single state."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.net.forward(state_t)
        return float(value.item())

    def save(self, path: str, step: int) -> None:
        """Save model, optimizer, and current step."""
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": step,
                "config": vars(self.cfg),
            },
            path,
        )

    def load(self, path: str) -> int:
        """Load model, optimizer, and return saved step."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return int(checkpoint.get("step", 0))

    def _compute_gae(self, memory: PPOMemory, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        rewards = np.array(memory.rewards, dtype=np.float32)
        values = np.array(memory.values + [last_value], dtype=np.float32)
        dones = np.array(memory.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * next_non_terminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(self, memory: PPOMemory, last_value: float, progress: float) -> Dict[str, float]:
        """Run one PPO update using the stored rollout."""
        cfg = self.cfg

        if cfg.anneal_lr:
            frac = 1.0 - progress
            lr_now = frac * cfg.learning_rate
            self.optimizer.param_groups[0]["lr"] = lr_now
        else:
            lr_now = self.optimizer.param_groups[0]["lr"]

        states = torch.tensor(np.array(memory.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(memory.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(memory.log_probs, dtype=torch.float32, device=self.device)
        advantages, returns = self._compute_gae(memory, last_value)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        values_old = torch.tensor(memory.values, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.shape[0]
        indices = np.arange(batch_size)

        clipfracs = []
        approx_kl = 0.0
        entropy_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0

        for _ in range(cfg.epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = indices[start:end]

                _, new_log_probs, entropy, new_values = self.net.get_action_and_value(states[mb_idx], actions[mb_idx])
                log_ratio = new_log_probs - old_log_probs[mb_idx]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_values = new_values.view(-1)
                v_loss_unclipped = (new_values - returns[mb_idx]) ** 2
                v_clipped = values_old[mb_idx] + torch.clamp(
                    new_values - values_old[mb_idx],
                    -cfg.clip_coef,
                    cfg.clip_coef,
                )
                v_loss_clipped = (v_clipped - returns[mb_idx]) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        return {
            "lr": float(lr_now),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_loss.item()),
            "approx_kl": float(approx_kl),
            "clipfrac": float(np.mean(clipfracs) if clipfracs else 0.0),
        }


# ============================================================
# Environment factory
# ============================================================

def make_env(render: bool = False, seed: int = SEED) -> gym.Env:
    """
    Create and wrap a CarRacing-v3 environment.

    The SAC.py file includes reward-related args, but PPO.py does not need
    SAC's reward wrapper. So only the overlapping, necessary workflow ideas
    were copied here. :contentReference[oaicite:1]{index=1}
    """
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    env = DiscreteCarRacingWrapper(env)
    env = CarRacingPreprocess(env)
    return env


# ============================================================
# Evaluation
# ============================================================

def evaluate(agent: PPOAgent, num_episodes: int = NUM_EVAL_EPS, seed: int = SEED) -> float:
    """Run evaluation and return mean reward."""
    eval_env = make_env(render=False, seed=seed + 100)
    total_reward = 0.0

    for ep in range(num_episodes):
        state, _ = eval_env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _, _ = agent.select_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward

        total_reward += ep_reward

    eval_env.close()
    return total_reward / num_episodes


# ============================================================
# CLI Argument Parser
# ============================================================

def parse_args():
    """
    Only includes CLI args that exist in the SAC.py file you sent,
    and only keeps the ones that are actually needed for PPO.py. :contentReference[oaicite:2]{index=2}
    """
    parser = argparse.ArgumentParser(description="PPO for CarRacing-v3")

    # Training / runtime args that exist in SAC.py
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)

    # Evaluation / rendering / resume args that exist in SAC.py
    parser.add_argument("--render", action="store_true", help="Render a trained agent using --checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt for render/eval mode.")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to .pt to resume training from.")
    parser.add_argument("--start-step", type=int, default=0, help="Initial step count (to match loaded checkpoint).")

    # Output arg that exists in SAC.py
    parser.add_argument("--save-dir", type=str, default="ppo_results",
                        help="Directory for checkpoints and logs.")

    return parser.parse_args()


# ============================================================
# Main Execution Loop
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, "progress.csv")
    latest_checkpoint_path = os.path.join(args.save_dir, "ppo_carracing.pt")

    maybe_init_csv(log_file)

    # Initialize environment and agent
    env = make_env(render=False, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    cfg = PPOConfig(total_timesteps=args.total_timesteps)
    agent = PPOAgent(env.observation_space.shape, env.action_space.n, cfg, DEVICE)

    # Rendering / evaluation mode
    if args.render:
        if args.checkpoint is None:
            print("[PPO] Error: Must provide --checkpoint to render.")
        else:
            print(f"[PPO] Rendering agent from {args.checkpoint}...")
            step = agent.load(args.checkpoint)
            print(f"[PPO] Loaded checkpoint at step {step}")

            eval_env = make_env(render=True, seed=args.seed)
            for ep in range(5):
                state, _ = eval_env.reset(seed=args.seed + ep)
                done = False
                total_r = 0.0
                ep_len = 0

                while not done:
                    action, _, _ = agent.select_action(state)
                    state, reward, terminated, truncated, _ = eval_env.step(action)
                    total_r += reward
                    ep_len += 1
                    done = terminated or truncated

                print(f"Episode {ep + 1}: Reward = {total_r:.2f} | Length = {ep_len}")

            eval_env.close()
        raise SystemExit

    # Resume training from checkpoint if requested
    global_step = args.start_step
    if args.load_checkpoint:
        print(f"[PPO] Loading checkpoint from {args.load_checkpoint} to resume training...")
        global_step = agent.load(args.load_checkpoint)

    print(f"[PPO] Starting training for {args.total_timesteps} steps (from step {global_step})...")
    print(f"[PPO] Device: {DEVICE}")

    memory = PPOMemory()
    state = obs

    start_time = time.time()
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    recent_rewards = collections.deque(maxlen=20)

    for t in range(global_step + 1, args.total_timesteps + 1):
        # Collect rollout data
        action, log_prob, value = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.add(state, action, log_prob, value, reward, done)

        state = next_state
        episode_reward += reward
        episode_length += 1

        # Episode end handling
        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1
            timeout_flag = info.get("grass_timeout", False)
            print(
                f"Episode {episode_count:4d} | "
                f"step {t:8d} | "
                f"reward {episode_reward:8.2f} | "
                f"length {episode_length:4d} | "
                f"grass_timeout={timeout_flag}"
            )
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        # PPO update once rollout buffer is full
        metrics = None
        if len(memory) >= cfg.rollout_steps:
            if memory.dones and memory.dones[-1]:
                last_value = 0.0
            else:
                last_value = agent.value(state)

            progress = t / max(1, cfg.total_timesteps)
            metrics = agent.update(memory, last_value, progress)
            memory.clear()

        # Evaluate, log CSV, and save checkpoint every eval_every steps
        if t % args.eval_every == 0:
            avg_eval_reward = evaluate(agent, num_episodes=NUM_EVAL_EPS, seed=args.seed)
            elapsed = time.time() - start_time
            fps = int((t - global_step) / elapsed) if elapsed > 0 else 0

            if metrics is not None:
                print(
                    f"[{t}/{args.total_timesteps}] "
                    f"Eval Reward: {avg_eval_reward:.2f} | "
                    f"FPS: {fps} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f}"
                )

                append_csv_row(
                    log_file,
                    [
                        t,
                        avg_eval_reward,
                        metrics["policy_loss"],
                        metrics["value_loss"],
                        metrics["entropy"],
                        metrics["approx_kl"],
                        metrics["clipfrac"],
                        metrics["lr"],
                    ],
                )
            else:
                print(f"[{t}/{args.total_timesteps}] Eval Reward: {avg_eval_reward:.2f} | FPS: {fps}")

                append_csv_row(
                    log_file,
                    [t, avg_eval_reward, 0, 0, 0, 0, 0, 0],
                )

            checkpoint_path = os.path.join(args.save_dir, f"ppo_carracing_{t}.pt")
            agent.save(checkpoint_path, t)
            agent.save(latest_checkpoint_path, t)
            print(f"[PPO] Saved checkpoints to {checkpoint_path} and {latest_checkpoint_path}")

    # Final save
    agent.save(latest_checkpoint_path, args.total_timesteps)
    print(f"[PPO] Training complete! Results saved to {args.save_dir}")
    env.close()