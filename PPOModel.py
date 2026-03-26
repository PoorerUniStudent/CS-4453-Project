import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


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
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

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
            checkpoint_dir (str): not actually used here, since self.checkpoint_file
                                  is already fixed above.
        """
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self, checkpoint_dir='tmp/ppo'):
        """
        Loads actor network weights from disk.

        Args:
            checkpoint_dir (str): not actually used here.

        Warning:
            This assumes the file already exists.
        """
        self.load_state_dict(T.load(self.checkpoint_file))
        
        
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
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

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
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self, checkpoint_dir='tmp/ppo'):
        """
        Loads critic network weights from disk.
        """
        self.load_state_dict(T.load(self.checkpoint_file))


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
        alpha=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        N=2048,
        n_epochs=10
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
        
    def save_models(self):
        """
        Saves both actor and critic parameters.
        """
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load_models(self):
        """
        Loads both actor and critic parameters.
        """
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        
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