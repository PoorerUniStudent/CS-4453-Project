"""
Proximal Policy Optimization (PPO) for Gymnasium CarRacing-v3
=============================================================
A PyTorch PPO implementation adapted to run on Gymnasium's
CarRacing-v3 environment using a DISCRETE action space.

This file keeps the original PPO agent implementation intact,
and adds:
  - top-level configurable hyperparameters
  - CarRacing observation preprocessing
  - environment factory
  - training / evaluation / rendering helpers
  - CLI argument parsing

Important:
  Your PPO actor uses a Categorical distribution, so this setup
  uses CarRacing-v3 with continuous=False.

Notes:
  - CarRacing observations are RGB images of shape (96, 96, 3).
  - Since the original PPO agent uses an MLP, we preprocess each
    frame into grayscale, flatten it, and feed the resulting vector
    into the agent.
  - This is not as strong as a CNN-based PPO, but it matches your
    current network architecture without changing the original core code.
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
LEARNING_RATE      = 5e-5       # Adam learning rate
GAMMA              = 0.99       # Discount factor
GAE_LAMBDA         = 0.95       # GAE lambda
POLICY_CLIP        = 0.2        # PPO clipping epsilon
BATCH_SIZE         = 64         # Mini-batch size
ROLLOUT_STEPS      = 4096       # Number of env steps before each PPO update
N_EPOCHS           = 10         # PPO training epochs per rollout
TOTAL_TIMESTEPS    = 300_000    # Total training steps
EVAL_EVERY         = 10_000     # Evaluate every N steps
NUM_EVAL_EPISODES  = 5          # Number of evaluation episodes
SEED               = 42         # Random seed

# Observation preprocessing
NORMALIZE_OBS      = True       # Normalize grayscale pixels to [0, 1]

# Output / checkpointing
SAVE_DIR           = "ppo_results"

# Rendering
RENDER_EPISODES    = 5


# ──────────────────────────────────────────────
# CUDA / MPS / CPU Device Selection
# ──────────────────────────────────────────────
if T.cuda.is_available():
    DEVICE = T.device("cuda")
elif hasattr(T.backends, "mps") and T.backends.mps.is_available():
    DEVICE = T.device("mps")
else:
    DEVICE = T.device("cpu")

print(f"[PPO] Using device: {DEVICE}")


# ──────────────────────────────────────────────
# Seeding
# ──────────────────────────────────────────────
def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
# Observation Preprocessing
# ──────────────────────────────────────────────
def preprocess_observation(obs: np.ndarray,
                           normalize: bool = NORMALIZE_OBS) -> np.ndarray:
    """
    Convert CarRacing RGB observation into a flattened grayscale vector.

    Steps:
      1. RGB -> grayscale
      2. Normalize to [0, 1] if requested
      3. Flatten into shape (96 * 96,)

    Args:
        obs: Raw CarRacing observation, shape (96, 96, 3)
        normalize: Whether to scale pixel values to [0, 1]

    Returns:
        1D float32 NumPy array.
    """
    # Convert RGB to grayscale using standard luminance weights
    gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

    if normalize:
        gray /= 255.0

    return gray.flatten().astype(np.float32)


def get_input_dims_from_env() -> tuple:
    """
    Returns the flattened observation shape expected by the PPO networks.
    """
    return (96 * 96,)


# ──────────────────────────────────────────────
# Reward Wrapper
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
        reward = self._shape_reward(reward, obs, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

    def _shape_reward(self, reward, obs, terminated, truncated, info):
        if self.mode == "default":
            return reward
        elif self.mode == "clip":
            return float(np.clip(reward, -1.0, 1.0))
        elif self.mode == "speed":
            # Car's true speed is rendered in the observation but not
            # directly exposed; we approximate via reward delta as a proxy.
            speed_bonus = 0.01 * max(reward - self._prev_reward, 0)
            self._prev_reward = reward
            return reward + speed_bonus
        elif self.mode == "custom":
            return self._custom_reward(reward, obs, terminated, truncated, info)
        else:
            raise ValueError(f"Unknown reward mode: '{self.mode}'")

    def _custom_reward(self, reward, obs, terminated, truncated, info):
        """Override this method in a subclass for fully custom rewards."""
        return reward


# ──────────────────────────────────────────────
# Environment Factory
# ──────────────────────────────────────────────
def make_env(render: bool = False, reward_mode: str = "default", seed: int = SEED):
    """
    Create a discrete CarRacing-v3 environment.

    Important:
      continuous=True gives a discrete action space, which matches your PPO
      implementation based on Categorical action sampling.
    """
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
    env = RewardWrapper(env, mode=reward_mode)
    return env


# ──────────────────────────────────────────────
# Original PPO Memory
# ──────────────────────────────────────────────
class PPOMemory:
    """
    Stores one rollout / trajectory of experience for PPO training.

    PPO does not usually learn from a single transition immediately.
    Instead, it collects many transitions:
        state, action, old log-prob, critic value, reward, done
    and then trains on batches from that stored data.

    Attributes:
        states      : list of observed states
        probs       : list of old log-probabilities of chosen actions
        vals        : list of critic value estimates V(s)
        actions     : list of actions taken
        rewards     : list of rewards received
        dones       : list of terminal flags (True if episode ended)
        batch_size  : number of samples per mini-batch during training
    """
    def __init__(self, batch_size):
        """
        Args:
            batch_size (int): size of each training mini-batch.
                              Common values: 32, 64, 128, 256
                              Must be >= 1.
        """
        self.states = []      # Stores environment observations (states)
        self.probs = []       # Stores log-probabilities of selected actions under old policy
        self.vals = []        # Stores critic estimates V(s) at the time of action selection
        self.actions = []     # Stores actions taken
        self.rewards = []     # Stores rewards from environment
        self.dones = []       # Stores whether each step ended an episode
        self.batch_size = batch_size

    def generate_batches(self):
        """
        Converts stored memory into NumPy arrays and creates shuffled mini-batches.

        Returns:
            states  (np.array): all stored states
            actions (np.array): all stored actions
            probs   (np.array): all stored old log-probabilities
            vals    (np.array): all stored state-value estimates
            rewards (np.array): all stored rewards
            dones   (np.array): all stored done flags
            batches (list[np.array]): list of arrays of shuffled indices for mini-batches

        Notes:
            - The total number of stored transitions is n_states.
            - Data is shuffled before batching, which helps training stability.
        """
        n_states = len(self.states)  # Total number of stored transitions

        # Starting indices of each batch: [0, batch_size, 2*batch_size, ...]
        batch_start = np.arange(0, n_states, self.batch_size)

        # Indices for all samples, e.g. [0, 1, 2, ..., n_states-1]
        indices = np.arange(n_states, dtype=np.int64)

        # Shuffle indices so batches are random
        np.random.shuffle(indices)

        # Slice shuffled indices into mini-batches
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        # Return everything as arrays for easier vectorized operations
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches
        )
        
    def store_memory(self, state, prob, val, action, reward, done):
        """
        Stores one transition in memory.

        Args:
            state  : observation/state from environment
            prob   : log-probability of selected action under current actor policy
            val    : critic estimate V(state)
            action : action taken by agent
            reward : reward received after taking action
            done   : whether this transition ended the episode

        Typical ranges:
            state  : depends entirely on environment
            prob   : log-probability, usually <= 0
            val    : any real number, depends on reward scale
            action : integer in [0, n_actions-1] for discrete action spaces
            reward : environment-specific, often small values like [-1, 1], but not always
            done   : boolean (True/False)
        """
        self.states.append(state)
        self.probs.append(prob)
        self.vals.append(val)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        """
        Clears all stored rollout data after PPO finishes training on it.
        """
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


# ──────────────────────────────────────────────
# Original Actor Network
# ──────────────────────────────────────────────
class ActorNetwork(nn.Module):
    """
    Policy network for PPO.

    This network takes a state as input and outputs a probability distribution
    over discrete actions.

    Architecture:
        state -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax

    Since Softmax is used, this actor is for DISCRETE action spaces only.
    """
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        """
        Args:
            n_actions (int): number of discrete actions available in the environment.
                             Must be >= 2 for most tasks.
            input_dims (tuple): shape of state input, e.g. (8,) or (4,)
            alpha (float): learning rate for Adam optimizer.
                           Common PPO range: 1e-5 to 3e-4
            fc1_dims (int): number of neurons in first hidden layer.
                            Common range: 64 to 512
            fc2_dims (int): number of neurons in second hidden layer.
                            Common range: 64 to 512
        """
        super(ActorNetwork, self).__init__()
        
        # File path where actor model parameters will be saved
        self.checkpoint_file = os.path.join('tmp/ppo', 'actor_torch_ppo')

        # Sequential actor network:
        # 1) input -> hidden layer
        # 2) ReLU activation
        # 3) hidden -> hidden
        # 4) ReLU activation
        # 5) hidden -> output logits
        # 6) Softmax converts logits to action probabilities summing to 1
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),  # input_dims is unpacked, e.g. (8,) -> 8
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        
        # Adam optimizer updates actor parameters using computed gradients
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Use GPU if available, else CPU
        self.device = DEVICE

        # Move model to chosen device
        self.to(self.device)

    def forward(self, state):
        """
        Runs a forward pass through the actor network.

        Args:
            state (Tensor): tensor of states, shape usually [batch_size, state_dim]

        Returns:
            dist (Categorical): categorical action distribution

        Notes:
            - self.actor(state) produces probabilities for each action.
            - Categorical(dist) wraps those probabilities into a distribution object.
            - Later we can sample actions and compute log-probabilities from it.
        """
        dist = self.actor(state)   # Action probabilities
        dist = Categorical(dist)   # Convert probabilities into a distribution
        
        return dist
    
    def save_checkpoint(self, checkpoint_dir='tmp/ppo'):
        """
        Saves actor network weights to disk.

        Args:
            checkpoint_dir (str): directory where checkpoint will be saved.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, 'actor_torch_ppo')
        T.save(self.state_dict(), save_path)
        
    def load_checkpoint(self, checkpoint_dir='tmp/ppo'):
        """
        Loads actor network weights from disk.

        Args:
            checkpoint_dir (str): directory where checkpoint was saved.
        """
        load_path = os.path.join(checkpoint_dir, 'actor_torch_ppo')
        self.load_state_dict(T.load(load_path, map_location=self.device))
        
        
# ──────────────────────────────────────────────
# Original Critic Network
# ──────────────────────────────────────────────
class CriticNetwork(nn.Module):
    """
    Value network for PPO.

    This network estimates the state-value function V(s), which is the expected
    future return from a given state.

    Architecture:
        state -> Linear -> ReLU -> Linear -> ReLU -> Linear -> scalar value
    """
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        """
        Args:
            input_dims (tuple): shape of state input
            alpha (float): learning rate for Adam optimizer
            fc1_dims (int): neurons in first hidden layer
            fc2_dims (int): neurons in second hidden layer
        """
        super(CriticNetwork, self).__init__()
        
        # File path where critic weights will be saved
        self.checkpoint_file = os.path.join('tmp/ppo', 'critic_torch_ppo')

        # Critic outputs a single scalar value for each state
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        
        # Optimizer for critic
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Use GPU if available
        self.device = DEVICE

        # Move model to selected device
        self.to(self.device)

    def forward(self, state):
        """
        Runs a forward pass through the critic.

        Args:
            state (Tensor): tensor of states

        Returns:
            value (Tensor): predicted state value V(s), shape [batch_size, 1]
        """
        value = self.critic(state)
        
        return value
    
    def save_checkpoint(self, checkpoint_dir='tmp/ppo'):
        """
        Saves critic network weights to disk.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, 'critic_torch_ppo')
        T.save(self.state_dict(), save_path)
        
    def load_checkpoint(self, checkpoint_dir='tmp/ppo'):
        """
        Loads critic network weights from disk.
        """
        load_path = os.path.join(checkpoint_dir, 'critic_torch_ppo')
        self.load_state_dict(T.load(load_path, map_location=self.device))


# ──────────────────────────────────────────────
# Original PPO Agent
# ──────────────────────────────────────────────
class PPOAgent:
    """
    Main PPO agent class.

    This class combines:
        - actor network (policy)
        - critic network (value function)
        - memory buffer
        - action selection
        - PPO learning update

    PPO is an on-policy algorithm, so the memory contains data generated by the
    current policy, and after learning, memory is cleared.
    """
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        policy_clip=POLICY_CLIP,
        batch_size=BATCH_SIZE,
        N=ROLLOUT_STEPS,
        n_epochs=N_EPOCHS
    ):
        """
        Args:
            n_actions (int): number of discrete actions
            input_dims (tuple): shape of observation/state
            alpha (float): learning rate
                           Typical PPO range: 1e-5 to 3e-4
            gamma (float): discount factor for future rewards
                           Typical range: 0.95 to 0.999
                           Must be in [0, 1]
            gae_lambda (float): lambda for Generalized Advantage Estimation
                                Typical range: 0.9 to 0.99
                                Must be in [0, 1]
            policy_clip (float): PPO clipping epsilon
                                 Typical range: 0.1 to 0.3
                                 Must be > 0
            batch_size (int): size of training mini-batches
                              Common: 32, 64, 128, 256
            N (int): rollout length / number of steps collected before learning
                     Common: 1024, 2048, 4096
                     Note: in this code, N is passed but never actually stored or used.
            n_epochs (int): number of times PPO trains over collected rollout data
                            Common: 3 to 10
        """
        self.alpha = alpha                    # Learning rate for optimizers
        self.gamma = gamma                    # Discount factor for future rewards
        self.gae_lambda = gae_lambda          # Controls bias/variance tradeoff in GAE
        self.policy_clip = policy_clip        # PPO clipping threshold
        self.n_epochs = n_epochs              # Number of epochs per PPO update
        
        self.actor = ActorNetwork(n_actions, input_dims, alpha)   # Policy network
        self.critic = CriticNetwork(input_dims, alpha)            # Value network
        self.memory = PPOMemory(batch_size)                       # Rollout memory
        
    def remember(self, state, prob, val, action, reward, done):
        """
        Stores one experience tuple in PPO memory.
        Usually called once per environment step.

        Args:
            state  : observation before action
            prob   : log-probability of chosen action
            val    : critic estimate of state value
            action : action selected
            reward : reward received
            done   : episode termination flag
        """
        self.memory.store_memory(state, prob, val, action, reward, done)
        
    def save_models(self, filepath):
        """
        Saves both actor and critic parameters into one .pt checkpoint file.
        """
        print('... saving model ...')
        T.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, filepath)
        
    def load_models(self, filepath):
        """
        Loads both actor and critic parameters from one .pt checkpoint file.
        """
        print('... loading model ...')
        filepath = os.path.join(SAVE_DIR, filepath)
        checkpoint = T.load(filepath, map_location=self.actor.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
    def choose_action(self, observation):
        """
        Selects an action using the current policy.

        Steps:
            1) Convert observation to tensor
            2) Get action distribution from actor
            3) Get state-value estimate from critic
            4) Sample an action from the distribution
            5) Return action, its log-probability, and value estimate

        Args:
            observation: environment state, usually a NumPy array

        Returns:
            action (int): chosen discrete action
            probs (float): log-probability of chosen action
            value (float): critic estimate V(s)

        Notes:
            - state is wrapped in [observation] so it becomes a batch of size 1
            - T.squeeze removes extra dimensions
        """
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        
        dist = self.actor(state)      # Policy distribution over actions
        value = self.critic(state)    # Estimated value of current state
        action = dist.sample()        # Sample one action
        
        probs = T.squeeze(dist.log_prob(action)).item()  # log pi(a|s)
        action = T.squeeze(action).item()                # convert action tensor -> Python int
        value = T.squeeze(value).item()                  # convert value tensor -> Python float
        
        return action, probs, value
        
    def learn(self):
        """
        Performs PPO training using all data currently stored in memory.

        Main steps:
            1) Get rollout data and shuffled batches
            2) Compute advantages using GAE-like calculation
            3) For each mini-batch:
                - compute new action log-probs
                - compute probability ratio
                - compute clipped actor loss
                - compute critic loss
                - backpropagate total loss
            4) Clear memory after training

        Important:
            PPO compares:
                new policy probability / old policy probability
            for the SAME actions from the rollout.

        Loss terms:
            actor_loss  : clipped PPO objective
            critic_loss : MSE between predicted value and target return
            total_loss  : actor_loss + 0.5 * critic_loss

        Note:
            This implementation does not include:
                - entropy bonus
                - normalization of advantages
                - explicit use of N
            which are often included in stronger PPO implementations.
        """
        for _ in range(self.n_epochs):
            # Retrieve stored rollout data and mini-batches
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            values = vals_arr  # Old critic predictions stored during rollout

            # Advantage array stores how much better/worse an action was than expected
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            # Compute advantage for each timestep
            # This is a backward-looking sum of TD errors with discounting
            for t in range(len(reward_arr) - 1):
                discount = 1   # cumulative discount factor: (gamma * lambda)^k
                a_t = 0        # advantage at timestep t

                # Accumulate future TD residuals from timestep t onward
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t
            
            # Convert advantages to tensor on same device as networks
            advantage = T.tensor(advantage).to(self.actor.device)
            
            # Convert stored values to tensor
            values = T.tensor(values).to(self.actor.device)
            
            # Train on each mini-batch
            for batch in batches:
                # Extract batch data and convert to tensors
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                
                # Forward pass through actor and critic with CURRENT parameters
                dist = self.actor(states)
                critic_value = self.critic(states)
                
                critic_value = critic_value.squeeze()  # shape [batch_size]
                
                # New log-probabilities of the same actions under updated policy
                new_probs = dist.log_prob(actions)

                # Probability ratio:
                # ratio = exp(new_log_prob) / exp(old_log_prob)
                #       = pi_new(a|s) / pi_old(a|s)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # Standard PPO surrogate objective
                weighted_probs = advantage[batch] * prob_ratio

                # Clipped surrogate objective
                weighted_clipped_probs = T.clamp(
                    prob_ratio,
                    1 - self.policy_clip,
                    1 + self.policy_clip
                ) * advantage[batch]

                # PPO actor loss is negative because we want to maximize objective
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Return target = advantage + old value estimate
                # Since A(s,a) = return - value, then return = A + V
                returns = advantage[batch] + values[batch]

                # Critic tries to predict returns
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                
                # Total loss: actor + scaled critic loss
                total_loss = actor_loss + 0.5 * critic_loss
                
                # Clear old gradients
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                # Backpropagate total loss
                total_loss.backward()

                # Update parameters
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        # PPO is on-policy, so after learning from rollout, clear it
        self.memory.clear_memory()


# ──────────────────────────────────────────────
# PPO + CarRacing Helpers
# ──────────────────────────────────────────────
def evaluate_agent(agent: PPOAgent,
                   num_episodes: int = NUM_EVAL_EPISODES,
                   reward_mode: str = "default",
                   seed: int = SEED) -> float:
    """
    Evaluate the PPO agent on CarRacing-v3.

    For simplicity, this uses the same stochastic policy sampling that your
    current choose_action() function uses.
    """
    eval_env = make_env(render=False, reward_mode=reward_mode, seed=seed + 100)
    total_reward = 0.0

    for ep in range(num_episodes):
        obs, info = eval_env.reset(seed=seed + 100 + ep)
        state = preprocess_observation(obs)

        done = False
        ep_reward = 0.0

        while not done:
            action, prob, val = agent.choose_action(state)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            state = preprocess_observation(next_obs)

            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward

    eval_env.close()
    return total_reward / num_episodes


def render_agent(agent: PPOAgent,
                 checkpoint_dir: str,
                 reward_mode: str = "default",
                 episodes: int = RENDER_EPISODES):
    """
    Render a trained PPO agent.
    """
    agent.load_models(checkpoint_dir)

    env = make_env(render=True, reward_mode=reward_mode)
    for ep in range(episodes):
        obs, info = env.reset()
        state = preprocess_observation(obs)
        done = False
        total_reward = 0.0

        while not done:
            action, prob, val = agent.choose_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = preprocess_observation(obs)
            total_reward += reward
            done = terminated or truncated

        print(f"[PPO] Render Episode {ep + 1}: Reward = {total_reward:.2f}")

    env.close()


def train_ppo_on_carracing(
    total_timesteps: int = TOTAL_TIMESTEPS,
    rollout_steps: int = ROLLOUT_STEPS,
    eval_every: int = EVAL_EVERY,
    reward_mode: str = "default",
    save_dir: str = SAVE_DIR,
    seed: int = SEED,
    alpha: float = LEARNING_RATE,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
    policy_clip: float = POLICY_CLIP,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = N_EPOCHS
):
    """
    Train the PPO agent on Gymnasium CarRacing-v3.

    This function:
      1. Creates the environment
      2. Preprocesses image observations into flat vectors
      3. Collects on-policy rollouts
      4. Calls agent.learn() every `rollout_steps`
      5. Periodically evaluates and saves the model
    """
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "progress.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("step,reward\n")

    env = make_env(render=False, reward_mode=reward_mode, seed=seed)
    obs, info = env.reset(seed=seed)
    state = preprocess_observation(obs)

    agent = PPOAgent(
        n_actions=env.action_space.n,
        input_dims=get_input_dims_from_env(),
        alpha=alpha,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        batch_size=batch_size,
        N=rollout_steps,
        n_epochs=n_epochs
    )

    print(f"[PPO] Starting training for {total_timesteps} steps...")
    print(f"[PPO] Device: {DEVICE} | Reward Mode: {reward_mode}")
    print(f"[PPO] Discrete action space size: {env.action_space.n}")
    print(f"[PPO] Input dims: {get_input_dims_from_env()}")

    start_time = time.time()
    episode_reward = 0.0
    episode_num = 1

    for step in range(1, total_timesteps + 1):
        # Choose action from current PPO policy
        action, prob, val = agent.choose_action(state)

        # Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess_observation(next_obs)

        done = terminated or truncated

        # Store transition in PPO memory
        agent.remember(state, prob, val, action, reward, done)

        state = next_state
        episode_reward += reward

        # PPO update after collecting enough rollout steps
        if step % rollout_steps == 0:
            agent.learn()

        # Handle episode termination
        if done:
            print(f"[PPO] Episode {episode_num} | Step {step} | Reward: {episode_reward:.2f}")
            obs, info = env.reset()
            state = preprocess_observation(obs)
            episode_reward = 0.0
            episode_num += 1

        # Periodic evaluation
        if step % eval_every == 0:
            avg_reward = evaluate_agent(
                agent,
                num_episodes=NUM_EVAL_EPISODES,
                reward_mode=reward_mode,
                seed=seed
            )

            elapsed = time.time() - start_time
            fps = int(step / elapsed) if elapsed > 0 else 0

            print(f"[PPO] [{step}/{total_timesteps}] Eval Reward: {avg_reward:.2f} | FPS: {fps}")

            with open(log_file, "a") as f:
                f.write(f"{step},{avg_reward}\n")

        # Periodic checkpoint
        if step % EVAL_EVERY == 0:
            checkpoint_dir = os.path.join(save_dir, f"ppo_car_{step}.pt")
            agent.save_models(checkpoint_dir)

    # Final save
    final_path = os.path.join(save_dir, "ppo_car_final.pt")
    agent.save_models(final_path)

    env.close()
    print(f"[PPO] Training complete! Results saved to {save_dir}")


# ──────────────────────────────────────────────
# CLI Argument Parser
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="PPO for CarRacing-v3 (Discrete)")

    # Training
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--alpha", type=float, default=LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
    parser.add_argument("--policy-clip", type=float, default=POLICY_CLIP)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)

    # Reward shaping
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="default",
        choices=["default", "clip"],
        help="Reward wrapper mode."
    )

    # Output
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)

    # Rendering / loading
    parser.add_argument("--render", action="store_true", help="Render a trained PPO agent.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a PPO .pt checkpoint file."
    )

    return parser.parse_args()


# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Render mode
    if args.render:
        if args.checkpoint is None:
            print("[PPO] Error: Must provide --checkpoint to render a trained model.")
        else:
            # Create a dummy env just to get action space size
            dummy_env = make_env(render=False, reward_mode=args.reward_mode, seed=args.seed)
            agent = PPOAgent(
                n_actions=dummy_env.action_space.n,
                input_dims=get_input_dims_from_env(),
                alpha=args.alpha,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                policy_clip=args.policy_clip,
                batch_size=args.batch_size,
                N=args.rollout_steps,
                n_epochs=args.n_epochs
            )
            dummy_env.close()

            render_agent(
                agent=agent,
                checkpoint_dir=args.checkpoint,
                reward_mode=args.reward_mode,
                episodes=RENDER_EPISODES
            )
    else:
        train_ppo_on_carracing(
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            eval_every=args.eval_every,
            reward_mode=args.reward_mode,
            save_dir=args.save_dir,
            seed=args.seed,
            alpha=args.alpha,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            policy_clip=args.policy_clip,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs
        )