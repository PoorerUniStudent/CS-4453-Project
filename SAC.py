"""
Soft Actor-Critic (SAC) for Gymnasium CarRacing-v3
===================================================
A from-scratch PyTorch implementation of SAC with:
  - CNN-based actor-critic for pixel observations
  - Grayscale + 4-frame stacking
  - Automatic entropy tuning
  - Configurable reward wrapper
  - Twin Q-networks to reduce overestimation bias

References:
  Haarnoja et al., 2018 — "Soft Actor-Critic: Off-Policy Maximum Entropy
  Deep Reinforcement Learning with a Stochastic Actor"
  https://doi.org/10.48550/arXiv.1801.01290
"""

import argparse
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
LEARNING_RATE    = 3e-4       # Adam LR for actor, critic, and alpha
GAMMA            = 0.99       # Discount factor
TAU              = 0.005      # Polyak averaging coefficient for target nets
ALPHA_INIT       = 0.2        # Initial entropy temperature
BUFFER_SIZE      = 50_000     # Reduced to 50k to fit in RAM (~1.8GB as uint8)
BATCH_SIZE       = 256        # Mini-batch size for gradient updates
LEARNING_STARTS  = 1_000      # Random exploration steps before training
TOTAL_TIMESTEPS  = 1_000_000  # Total environment interactions
EVAL_EVERY       = 10_000     # Evaluate every N steps
NUM_EVAL_EPS     = 5          # Episodes per evaluation round
FRAME_STACK      = 4          # Number of consecutive frames to stack
SEED             = 42         # Random seed for reproducibility

# ──────────────────────────────────────────────
# CUDA(Nvidia) → MPS(Apple Silicon) → CPU Fallback
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[SAC] Using device: {DEVICE}")

# ──────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────
def set_seed(seed: int):
    """Set random seeds for reproducibility across numpy, torch, and gym."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# Observation Pre-processing
# ──────────────────────────────────────────────
def preprocess_frame(obs: np.ndarray) -> np.ndarray:
    """
    Convert a single 96×96×3 RGB frame to a 96×96 grayscale float32 array
    with values normalized to [0, 1].
    """
    # Standard luminance weights: 0.2989 R + 0.5870 G + 0.1140 B
    gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
    gray /= 255.0
    return gray  # shape: (96, 96)


class FrameStack:
    """
    Maintains a stack of the last `n` preprocessed frames.
    Provides velocity/acceleration cues that a single frame cannot.
    """

    def __init__(self, n: int = FRAME_STACK):
        self.n = n
        self._frames: deque = deque(maxlen=n)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Clear the stack and fill it with copies of the initial frame."""
        frame = preprocess_frame(obs)
        self._frames.clear()
        for _ in range(self.n):
            self._frames.append(frame)
        return self._get_stack()

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Push a new frame and return the updated stack."""
        self._frames.append(preprocess_frame(obs))
        return self._get_stack()

    def _get_stack(self) -> np.ndarray:
        """Return stacked frames as a (n, 96, 96) float32 array."""
        return np.array(self._frames, dtype=np.float32)


# ──────────────────────────────────────────────
# Configurable Reward Wrapper
# ──────────────────────────────────────────────
class RewardWrapper(gym.Wrapper):
    """
    Wraps CarRacing to allow experimenting with different reward signals.

    Modes:
      - "default"    : Use the environment's built-in reward unchanged.
      - "clip"       : Clip rewards to [-1, +1] for training stability.
      - "speed"      : Add a small bonus proportional to the car's speed,
                       encouraging the agent to drive faster.
      - "custom"     : Override `custom_reward()` for your own design.

    Allows studying of reward shaping affects of driving behavior,
    as described in the project proposal.
    """

    def __init__(self, env: gym.Env, mode: str = "default"):
        super().__init__(env)
        self.mode = mode
        self._prev_reward = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Store raw reward in info for later analysis
        info['raw_reward'] = float(reward)
        shaped_reward = self._shape_reward(reward, obs, terminated, truncated, info)
        return obs, float(shaped_reward), terminated, truncated, info

    def _shape_reward(self, reward, obs, terminated, truncated, info):
        if self.mode == "default":
            return reward
        elif self.mode == "clip":
            return float(np.clip(reward, -1.0, 1.0))
        elif self.mode == "oldspeed": # first speed run reward
            # Car's true speed is rendered in the observation but not
            # directly exposed; we approximate via reward delta as a proxy.
            speed_bonus = 0.01 * max(reward - self._prev_reward, 0)
            self._prev_reward = reward
            return reward + speed_bonus
        elif self.mode == "speed":
            # Small bonus for velocity magnitude, only if making progress.
            making_progress = reward > 0
            vel = self.env.unwrapped.car.hull.linearVelocity
            speed = np.sqrt(np.sum(np.square(vel)))
            speed_bonus = 0.1 * speed if making_progress else 0.0
            return reward + speed_bonus
        elif self.mode == "precision":
            # 1. Base reward
            # 2. Speed bonus (forward only, progress-gated)
            # 3. Spinning penalty (angular velocity)
            # 4. Backward sliding penalty
            car = self.env.unwrapped.car
            vx, vy = car.hull.linearVelocity
            angle = car.hull.angle
            speed = np.sqrt(vx**2 + vy**2)
            
            # Unit vector of the car's nose (Box2D CarRacing specific)
            unit_x, unit_y = -np.sin(angle), np.cos(angle)
            forward_speed = vx * unit_x + vy * unit_y
            
            # Progress Gate: Only reward speed if actually hitting NEW tiles
            making_progress = reward > 0
            
            # Bonuses and Penalties
            speed_bonus = 0.1 * max(forward_speed, 0) if making_progress else 0.0
            spin_penalty = 0.5 * abs(car.hull.angularVelocity)
            backward_penalty = 5.0 if forward_speed < -1.0 else 0.0
            
            # Fail-Safe: Penalty for moving fast but NOT making progress (wrong way/off-track)
            progress_penalty = 1.0 if (not making_progress and speed > 5.0) else 0.0
            
            return reward + speed_bonus - spin_penalty - backward_penalty - progress_penalty
        elif self.mode == "custom":
            return self._custom_reward(reward, obs, terminated, truncated, info)
        else:
            raise ValueError(f"Unknown reward mode: '{self.mode}'")

    def _custom_reward(self, reward, obs, terminated, truncated, info):
        """Override this method in a subclass for fully custom rewards."""
        return reward


# ──────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────
class ReplayBuffer:
    """
    Fixed-size buffer to store transition tuples.
    Transitions: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int, state_shape: tuple, action_dim: int):
        self.capacity = capacity
        # Use uint8 for states to save 4x memory (1 byte vs 4 bytes per pixel)
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Store a new transition in the buffer (converts to uint8)."""
        # Convert 0.0-1.0 float to 0-255 uint8
        self.states[self.index] = (state * 255).astype(np.uint8)
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = (next_state * 255).astype(np.uint8)
        self.dones[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Randomly sample a batch of transitions as PyTorch tensors (converts back to float32)."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        # Convert back to 0.0-1.0 float32 on the GPU/CPU
        s = torch.as_tensor(self.states[idxs], device=DEVICE).float() / 255.0
        a = torch.as_tensor(self.actions[idxs], device=DEVICE)
        r = torch.as_tensor(self.rewards[idxs], device=DEVICE)
        ns = torch.as_tensor(self.next_states[idxs], device=DEVICE).float() / 255.0
        d = torch.as_tensor(self.dones[idxs], device=DEVICE)

        return s, a, r, ns, d


# ──────────────────────────────────────────────
# Neural Network Architectures
# ──────────────────────────────────────────────
class CNNFeatureExtractor(nn.Module):
    """
    Nature DQN-style CNN to process 96x96 images.
    Shared by both the Actor and the Critic.
    """

    def __init__(self, input_channels: int = FRAME_STACK, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape after convolution:
        # 1. (96-8)/4 + 1 = 23
        # 2. (23-4)/2 + 1 = 10
        # 3. (10-3)/1 + 1 = 8
        # Flattened features: 64 * 8 * 8 = 4096
        self.fc = nn.Sequential(
            nn.Linear(4096, feature_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class SoftQNetwork(nn.Module):
    """
    Critic: Estimates Q(s, a).
    Takes state features + action as input.
    """

    def __init__(self, feature_dim: int = 512, action_dim: int = 3):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, action], dim=1)
        return self.q_net(x)


class ActorNetwork(nn.Module):
    """
    Actor: Outputs a Gaussian distribution over actions.
    Uses reparameterization trick for differentiable sampling.
    """

    def __init__(self, feature_dim: int = 512, action_dim: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        # Standard log_std bounds to prevent numerical instability
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, features: torch.Tensor):
        x = self.fc(features)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state_features: torch.Tensor):
        """
        Samples action from Gaussian + squashes with Tanh.
        Returns: (action, log_prob)
        """
        mu, log_std = self.forward(state_features)
        std = log_std.exp()

        # Reparameterization trick: a = mu + std * epsilon
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # Enforce action limits [steering: -1/1, gas: 0/1, brake: 0/1]
        # CarRacing-v3 obs space is symmetric [-1, 1], so we handle
        # environment-specific bounds via rescaling if needed, but
        # Tanh is already [-1, 1].
        action = y_t

        # Correction for Tanh squashing in log-probability calculation
        # log π(a|s) = log μ(x|s) - sum(log(1 - tanh²(x_i)))
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def get_action(self, state_features: torch.Tensor, deterministic: bool = False):
        """Return the action (mean if deterministic, sample if stochastic)."""
        mu, log_std = self.forward(state_features)
        if deterministic:
            return torch.tanh(mu)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mu, std)
            return torch.tanh(normal.rsample())


# ──────────────────────────────────────────────
# SAC Agent
# ──────────────────────────────────────────────
class SACAgent:
    """
    The core SAC algorithm implementation.
    Manages the actor, twin critics, target networks, and automatic entropy tuning.
    """

    def __init__(self, action_dim: int = 3, feature_dim: int = 512, lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, min_alpha: float = None, alpha_lr: float = 3e-4):
        self.gamma = gamma
        self.tau = tau
        self.alpha_lr = alpha_lr
        self.min_log_alpha = np.log(min_alpha) if min_alpha is not None else None

        # Networks
        self.feature_extractor = CNNFeatureExtractor(FRAME_STACK, feature_dim).to(DEVICE)
        self.actor = ActorNetwork(feature_dim, action_dim).to(DEVICE)
        self.critic1 = SoftQNetwork(feature_dim, action_dim).to(DEVICE)
        self.critic2 = SoftQNetwork(feature_dim, action_dim).to(DEVICE)

        # Target Networks (only for critics in SAC)
        self.target_critic1 = SoftQNetwork(feature_dim, action_dim).to(DEVICE)
        self.target_critic2 = SoftQNetwork(feature_dim, action_dim).to(DEVICE)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        # Note: We include feature_extractor in BOTH actor and critic updates,
        # or we could keep it separate. Here, we'll optimize it with the critics.
        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters()) + list(self.feature_extractor.parameters())
        self.critic_optimizer = optim.Adam(self.critic_params, lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Automatic Entropy Tuning
        self.target_entropy = -action_dim
        # Initial log_alpha (using -1.6 for exp(-1.6) ~= 0.2 initial alpha)
        # Note: Must be a leaf tensor for the optimizer
        self.log_alpha = torch.tensor([-1.6], requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Convert state to tensor and get action from actor."""
        state = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(state)
            action = self.actor.get_action(features, deterministic)
        return action.cpu().numpy()[0]

    def update(self, buffer: ReplayBuffer, batch_size: int):
        """Sample a batch and update all networks."""
        s, a, r, ns, d = buffer.sample(batch_size)

        # 1. Update Critics
        with torch.no_grad():
            # Get next features and next actions
            next_features = self.feature_extractor(ns)
            next_action, next_log_prob = self.actor.sample(next_features)

            # Compute target Q-values using Twin-Q and Target Critics
            q1_next = self.target_critic1(next_features, next_action)
            q2_next = self.target_critic2(next_features, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob

            # Bellman Equation: y = r + gamma * (1-d) * Q_next
            target_q = r + self.gamma * (1 - d) * q_next

        # Current Q-estimates
        curr_features = self.feature_extractor(s)
        q1 = self.critic1(curr_features, a)
        q2 = self.critic2(curr_features, a)

        # Critic Loss (MSE)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Update Actor
        # Re-extract features (since feature_extractor was just updated)
        # For correctness with the gradient flow, we re-sample from the actor.
        curr_features_actor = self.feature_extractor(s).detach()
        new_action, log_prob = self.actor.sample(curr_features_actor)

        q1_new = self.critic1(curr_features_actor, new_action)
        q2_new = self.critic2(curr_features_actor, new_action)
        q_new = torch.min(q1_new, q2_new)

        # Actor Loss: minimize (alpha * log_prob - Q)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Update Alpha (Entropy Temperature)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Clamp log_alpha if a minimum is provided
        if self.min_log_alpha is not None:
            with torch.no_grad():
                self.log_alpha.clamp_(min=self.min_log_alpha)

        # 4. Soft Update Target Networks
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item()
        }

    def set_lr(self, lr: float):
        """Manually update the learning rate for all optimizers."""
        for opt in [self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer]:
            for param_group in opt.param_groups:
                param_group['lr'] = lr

    def toggle_feature_extractor(self, enabled: bool):
        """Enable or disable gradients for the shared CNN feature extractor."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = enabled

    def save(self, path: str):
        """Save network weights, optimizers, and alpha."""
        torch.save({
            "feature_extractor": self.feature_extractor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "log_alpha": self.log_alpha,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load network weights and optimizer states."""
        ckpt = torch.load(path, map_location=DEVICE)
        self.feature_extractor.load_state_dict(ckpt["feature_extractor"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Load alpha and optimizers if they exist in the checkpoint
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"].data)
        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        if "alpha_optimizer" in ckpt:
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])


# ──────────────────────────────────────────────
# Environment Factory
# ──────────────────────────────────────────────
def make_env(render: bool = False, reward_mode: str = "default", seed: int = SEED):
    """
    Create and wrap a CarRacing-v3 environment.

    Args:
        render:      If True, create with human-visible rendering.
        reward_mode: Reward wrapper mode ("default", "clip", "speed", "custom").
        seed:        Random seed for the environment.

    Returns:
        Wrapped Gymnasium environment.
    """
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    env = RewardWrapper(env, mode=reward_mode)
    return env


def evaluate(agent: SACAgent, reward_mode: str, num_episodes: int = NUM_EVAL_EPS, seed: int = SEED):
    """Run `num_episodes` using the deterministic policy and return (mean_shaped, mean_raw)."""
    eval_env = make_env(render=False, reward_mode=reward_mode, seed=seed + 100)
    total_reward_shaped = 0.0
    total_reward_raw = 0.0

    for i in range(num_episodes):
        obs, info = eval_env.reset(seed=seed + i)
        frame_stack = FrameStack(n=FRAME_STACK)
        stacked = frame_stack.reset(obs)
        done = False
        ep_reward_shaped = 0.0
        ep_reward_raw = 0.0

        while not done:
            action = agent.select_action(stacked, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            stacked = frame_stack.step(obs)
            ep_reward_shaped += reward
            ep_reward_raw += info.get('raw_reward', reward)
            done = terminated or truncated

        total_reward_shaped += ep_reward_shaped
        total_reward_raw += ep_reward_raw

    eval_env.close()
    return total_reward_shaped / num_episodes, total_reward_raw / num_episodes


# ──────────────────────────────────────────────
# CLI Argument Parser
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="SAC for CarRacing-v3")

    # Training
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--seed", type=int, default=SEED)

    # Reward shaping
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="default",
        choices=["default", "clip", "oldspeed", "speed", "precision", "custom"],
        help="Reward wrapper mode for experimentation.",
    )

    # Evaluation / rendering / resume
    parser.add_argument("--render", action="store_true", help="Render a trained agent using --checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt for --render mode.")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to .pt to resume training from.")
    parser.add_argument("--start-step", type=int, default=0, help="Initial step count (to match loaded checkpoint).")
    parser.add_argument("--refill-steps", type=int, default=1000, help="Number of random steps to refill buffer when resuming.")
    parser.add_argument("--alpha-init", type=float, default=None, help="Initial alpha (overrides default or checkpoint).")
    parser.add_argument("--resume-lr", type=float, default=None, help="Starting LR for resume (stable fallback).")
    parser.add_argument("--warmup-steps", type=int, default=50000, help="Steps to linearly increase LR back to original.")
    parser.add_argument("--freeze-cnn-until", type=int, default=None, help="Step count until which the CNN remains frozen.")
    parser.add_argument("--min-alpha", type=float, default=None, help="Minimum alpha floor to prevent entropy collapse.")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="Learning rate for entropy temperature tuning.")

    # Output
    parser.add_argument("--save-dir", type=str, default="sac_results",
                        help="Directory for checkpoints and logs.")

    return parser.parse_args()


# ──────────────────────────────────────────────
# Main Execution Loop
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, "progress.csv")

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("step,reward_shaped,reward_raw,critic_loss,actor_loss,alpha\n")

    # Initialize environment and components
    env = make_env(render=False, reward_mode=args.reward_mode, seed=args.seed)
    obs, info = env.reset(seed=args.seed)

    frame_stack = FrameStack(n=FRAME_STACK)
    stacked = frame_stack.reset(obs)

    agent = SACAgent(
        action_dim=env.action_space.shape[0],
        feature_dim=512,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        min_alpha=args.min_alpha,
        alpha_lr=args.alpha_lr
    )

    # Handle Rendering / Evaluation Mode
    if args.render:
        if args.checkpoint is None:
            print("[SAC] Error: Must provide --checkpoint to render.")
        else:
            print(f"[SAC] Rendering agent from {args.checkpoint}...")
            print(f"[SAC] Using Reward Mode: {args.reward_mode}")
            agent.load(args.checkpoint)
            eval_env = make_env(render=True, reward_mode=args.reward_mode)
            for ep in range(5):
                obs, _ = eval_env.reset(seed=args.seed + ep)
                fs = FrameStack(n=FRAME_STACK)
                stk = fs.reset(obs)
                done = False
                total_r_shaped = 0
                total_r_raw = 0
                while not done:
                    act = agent.select_action(stk, deterministic=True)
                    obs, r, term, trunc, info = eval_env.step(act)
                    stk = fs.step(obs)
                    total_r_shaped += r
                    total_r_raw += info.get('raw_reward', r)
                    done = term or trunc
                print(f"Episode {ep+1}: Shaped Reward = {total_r_shaped:.2f} | Raw Reward = {total_r_raw:.2f}")
            eval_env.close()
        exit()

    buffer = ReplayBuffer(args.buffer_size, stacked.shape, env.action_space.shape[0])

    if args.load_checkpoint:
        # Load weights for actor/critic to continue training from a saved state
        print(f"[SAC] Loading checkpoint from {args.load_checkpoint} to resume training...")
        agent.load(args.load_checkpoint)

    # Allow overriding alpha if provided in CLI (useful for fixing old checkpoints)
    if args.alpha_init is not None:
        print(f"[SAC] Overriding Alpha to {args.alpha_init}")
        with torch.no_grad():
            agent.log_alpha.data.fill_(np.log(args.alpha_init))

    print(f"[SAC] Starting training for {args.total_timesteps} steps (from step {args.start_step})...")
    print(f"[SAC] Device: {DEVICE} | Reward Mode: {args.reward_mode}")

    start_time = time.time()
    ep_reward = 0
    ep_length = 0

    target_lr = args.lr
    current_lr = target_lr

    # Use start_step as the initial value for the counter
    for t in range(args.start_step + 1, args.total_timesteps + 1):
        # 0a. CNN Freezing (Vision Protection)
        if args.freeze_cnn_until is not None:
            if t <= args.freeze_cnn_until:
                agent.toggle_feature_extractor(False)
            else:
                agent.toggle_feature_extractor(True)

        # 0b. Learning Rate Warmup/Recovery
        # If we are in the warmup phase after a resume, adjust the LR
        if args.resume_lr is not None and t > args.start_step + args.refill_steps:
            relative_step = t - (args.start_step + args.refill_steps)
            progress = min(1.0, relative_step / args.warmup_steps)
            new_lr = args.resume_lr + progress * (target_lr - args.resume_lr)
            if new_lr != current_lr:
                agent.set_lr(new_lr)
                current_lr = new_lr

        # 1. Action selection
        # If we are starting from scratch OR we just resumed and the buffer is still refilling
        if t < args.learning_starts or buffer.size < args.refill_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(stacked)

        # 2. Environment step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_stacked = frame_stack.step(next_obs)
        ep_reward += reward
        ep_length += 1

        # 3. Store transition
        # We handle 'truncated' as not a terminal state for the Value function bootstrap
        buffer.add(stacked, action, reward, next_stacked, terminated)
        stacked = next_stacked

        # 4. Update iteration
        metrics = None
        # Only start training if we are past learning_starts AND the buffer has refilled
        if t >= args.learning_starts and buffer.size >= args.refill_steps and buffer.size >= args.batch_size:
            metrics = agent.update(buffer, args.batch_size)

        # 5. Handle episode end
        if terminated or truncated:
            obs, info = env.reset()
            stacked = frame_stack.reset(obs)
            ep_reward = 0
            ep_length = 0

        # 6. Evaluation and Logging
        if t % args.eval_every == 0:
            avg_reward_shaped, avg_reward_raw = evaluate(agent, args.reward_mode, seed=args.seed)
            # Use relative time and steps for FPS calculation
            elapsed = time.time() - start_time
            fps = int((t - args.start_step) / elapsed) if elapsed > 0 else 0

            print(f"[{t}/{args.total_timesteps}] "
                  f"Eval Shaped: {avg_reward_shaped:.2f} | Raw: {avg_reward_raw:.2f} | "
                  f"FPS: {fps} | "
                  f"Alpha: {metrics['alpha']:.4f}" if metrics else "")

            # Save progress
            with open(log_file, "a") as f:
                c_loss = metrics['critic_loss'] if metrics else 0
                a_loss = metrics['actor_loss'] if metrics else 0
                alpha = metrics['alpha'] if metrics else 0
                f.write(f"{t},{avg_reward_shaped},{avg_reward_raw},{c_loss},{a_loss},{alpha}\n")

            # Save checkpoint
            checkpoint_path = os.path.join(args.save_dir, f"sac_car_{t}.pt")
            agent.save(checkpoint_path)

    print(f"[SAC] Training complete! Results saved to {args.save_dir}")
    env.close()
