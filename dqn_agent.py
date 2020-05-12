import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

A = 0.6
B_GROWTH_RATE = 0.005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double_dqn=False, priority_replay=False, dueling_network=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.double_dqn = double_dqn
        self.priority_replay = priority_replay
        self.dueling_network = dueling_network

        # Q-Network
        if self.dueling_network:
            self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if self.priority_replay:
            self.memory = PrioritizedReplayBuffer(state_size, BUFFER_SIZE, BATCH_SIZE, seed, use_rank=True)
        else:
            self.memory = ReplayBuffer(state_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        if self.priority_replay:
            # calculate current td error to determine priority
            torch_state = torch.from_numpy(state).reshape((1,state.shape[0])).float().to(device)
            torch_next_state = torch.from_numpy(next_state).reshape((1,next_state.shape[0])).float().to(device)
            self.qnetwork_local.eval()
            self.qnetwork_target.eval()
            with torch.no_grad():
                q_values = self.qnetwork_local(torch_state).cpu().data.numpy()[0]
                q_values_prime = self.qnetwork_local(torch_next_state).cpu().data.numpy()[0]
                target_q_values = self.qnetwork_target(torch_next_state).cpu().data.numpy()[0]
            self.qnetwork_local.train()
            self.qnetwork_target.train()

            td_error = reward + GAMMA*target_q_values[np.argmax(q_values_prime)]*(1-done) - q_values[action]
            priority = abs(td_error) + 0.01

            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done, priority)
        else:
            # Save experience in replay memory
            self.memory.add(state, action, reward, next_state, done)
            
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.action_size)).astype(int)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """        
        if self.priority_replay:
            states, actions, rewards, next_states, dones, priorities, experiences_idx = experiences
            # Calculate Importance Sampling weight and update beta parameter
            sampling_weight = (1/len(self.memory))*(1/priorities)**self.B
            sampling_weight = (sampling_weight/torch.max(sampling_weight)).reshape((sampling_weight.shape[0],1))
            self.B = min(self.B+B_GROWTH_RATE, 1)
        else:
            states, actions, rewards, next_states, dones = experiences
        
        states = states.to(device)
        next_states = next_states.to(device)
        
        # Get q values of chosen actions
        local_q_values = self.qnetwork_local(states)
        chosen_q_values = torch.gather(local_q_values, 1, actions)
        
        # Double DQN Implementation
        if self.double_dqn:
            # get next actions based on local network q values
            # evaluate next actions based on target network q values
            next_q_values = self.qnetwork_local(next_states).detach()
            target_q_values = self.qnetwork_target(next_states).detach()
            greedy_actions = torch.max(next_q_values, 1)[1].reshape((next_q_values.shape[0],1))
            greedy_q_values = torch.gather(target_q_values, 1, greedy_actions)
        else:
            # get next actions based on local network q values
            # evaluate next actions based on local network q values
            target_q_values = self.qnetwork_target(next_states).detach()
            greedy_q_values = torch.max(target_q_values, 1)[0].reshape((target_q_values.shape[0],1))
            
        # calculate TD Target 
        td_target = rewards + gamma*greedy_q_values*(1-dones)
        
        if self.priority_replay:
            # calculate loss with importance sampling weight
            td_error = td_target - chosen_q_values
            weighted_td_error = sampling_weight*td_error.pow(2)
            loss = weighted_td_error.mean()
            
            # update priorities for chosen experiences
            abs_td_error = (torch.abs(td_error) + 0.01).cpu().data.numpy()
            self.memory.update_priorities(experiences_idx, abs_td_error)
        else:
            # calculate loss
            loss = F.mse_loss(chosen_q_values, td_target)   
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_size (int): dimension of state space
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.state_memory = np.zeros((buffer_size, state_size))
        self.action_memory = np.zeros((buffer_size, 1))
        self.reward_memory = np.zeros((buffer_size, 1))
        self.next_state_memory = np.zeros((buffer_size, state_size))
        self.done_memory = np.zeros((buffer_size, 1))
        
        self.buffer_size = buffer_size
        self.current_buffer_pointer = 0
        self.num_items = 0
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.state_memory[self.current_buffer_pointer] = state
        self.action_memory[self.current_buffer_pointer] = action
        self.reward_memory[self.current_buffer_pointer] = reward
        self.next_state_memory[self.current_buffer_pointer] = next_state
        self.done_memory[self.current_buffer_pointer] = done
        
        self.current_buffer_pointer = (self.current_buffer_pointer + 1) % self.buffer_size
        if self.num_items < self.buffer_size:
            self.num_items += 1
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences_idx = np.random.choice(self.num_items, self.batch_size, False)

        states = torch.from_numpy(self.state_memory[experiences_idx,:]).float().to(device)
        actions = torch.from_numpy(self.action_memory[experiences_idx,:]).long().to(device)
        rewards = torch.from_numpy(self.reward_memory[experiences_idx,:]).float().to(device)
        next_states = torch.from_numpy(self.next_state_memory[experiences_idx,:]).float().to(device)
        dones = torch.from_numpy(self.done_memory[experiences_idx,:]).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.num_items
    
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, buffer_size, batch_size, seed, use_rank=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_size (int): dimension of each state space
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.use_rank = use_rank
        
        self.state_memory = np.zeros((buffer_size, state_size))
        self.action_memory = np.zeros((buffer_size, 1))
        self.reward_memory = np.zeros((buffer_size, 1))
        self.next_state_memory = np.zeros((buffer_size, state_size))
        self.done_memory = np.zeros((buffer_size, 1))
        self.priority_memory = np.zeros((buffer_size, 1)) - 1
        
        self.buffer_size = buffer_size
        self.current_buffer_pointer = 0
        self.num_items = 0
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        self.state_memory[self.current_buffer_pointer] = state
        self.action_memory[self.current_buffer_pointer] = action
        self.reward_memory[self.current_buffer_pointer] = reward
        self.next_state_memory[self.current_buffer_pointer] = next_state
        self.done_memory[self.current_buffer_pointer] = done
        self.priority_memory[self.current_buffer_pointer] = priority
        
        self.current_buffer_pointer = (self.current_buffer_pointer + 1) % self.buffer_size
        if self.num_items < self.buffer_size:
            self.num_items += 1
    def update_priorities(self, priorities_indices, priorities):
        """Update existing priorites"""
        self.priority_memory[priorities_indices] = priorities;
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.use_rank:
            # calculate rank-based prioirtes
            priorities = self.priority_memory[:self.num_items].reshape((self.num_items,))
            sorted_idx = np.argsort(priorities)
            ranks = np.arange(sorted_idx.shape[0], 0, -1)
            ranks = (1/ranks)**A
            priority_list = (ranks/np.sum(ranks)).reshape((ranks.shape[0],))
            # choose experiences based on rank-based priorities
            experiences_idx = np.random.choice(sorted_idx, self.batch_size, False, priority_list)
        else:
            # calculate proportional priorities
            priorities = self.priority_memory[:self.num_items]
            priorities = priorities**A
            priority_list = (priorities/np.sum(priorities)).reshape((priorities.shape[0],))
            # choose experiences based on proportional priorities
            experiences_idx = np.random.choice(self.num_items, self.batch_size, False, priority_list)

        states = torch.from_numpy(self.state_memory[experiences_idx,:]).float().to(device)
        actions = torch.from_numpy(self.action_memory[experiences_idx,:]).long().to(device)
        rewards = torch.from_numpy(self.reward_memory[experiences_idx,:]).float().to(device)
        next_states = torch.from_numpy(self.next_state_memory[experiences_idx,:]).float().to(device)
        dones = torch.from_numpy(self.done_memory[experiences_idx,:]).float().to(device)
        priorities = torch.from_numpy(priority_list[experiences_idx]).float().to(device)
  
        return (states, actions, rewards, next_states, dones, priorities, experiences_idx)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.num_items