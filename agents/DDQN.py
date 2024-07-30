from collections import deque
import random

import numpy as np
import torch
from torch import nn


class Neural_Network(nn.Module):
    def __init__(self, actions, state_size, width):
        super(Neural_Network, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_size, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, actions))

    def forward(self, state):
        return self.net(state)


class DDQNAgent:
    def __init__(self, state_size, action_size, buffer_size, epsilon, epsilon_decay, learning_rate, NN_width,
                 epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.network = Neural_Network(action_size, state_size, NN_width)
        self.target_network = Neural_Network(action_size, state_size, NN_width)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.network(state)
                best_action = torch.argmax(q_values).item()
                return best_action

    def add_to_buffer(self, state, action, reward, state_prime, terminal):
        self.buffer.append((state, action, reward, state_prime, terminal))

    def update_network(self):
        samples = self.sample_buffer()
        states, actions, rewards, states_primes, terminals = zip(*samples)

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def sample_buffer(self):
        max_size = min(len(self.buffer), self.buffer_size)
        return random.sample(self.buffer, max_size)
