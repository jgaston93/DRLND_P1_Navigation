import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,action_size)
        
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        q_values = self.fc3(x)
            
        return q_values
    
class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128,128)
        
        # used to calculate advantage values
        self.advantage = nn.Linear(128,action_size)
        
        # used to calculate state values
        self.value = nn.Linear(128,1)
        
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        # get output from advantage portion of the network
        advantage_values = self.advantage(x)
        
        # get output from the value porition of the network
        state_value = self.value(x)
        
        # combine those back together using equation 9 from dueling dqn paper
        # Dueling Network Architectures for Deep Reinforcement Learning
        q_values = state_value + (advantage_values - torch.mean(advantage_values,1).reshape((advantage_values.shape[0],1)))
            
        return q_values

    
    