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
    def __init__(self, state_size, action_size, buffer_size, learning_rate, NN_width,
                 gamma, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.gamma = gamma
        self.learning_rate = learning_rate

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.network = Neural_Network(action_size, state_size, NN_width).to(self.device)
        self.target_network = Neural_Network(action_size, state_size, NN_width).to(self.device)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().to(self.device) / 255.0
                q_values = self.network(state)
                best_action = torch.argmax(q_values).item()
                return best_action

    def add_to_buffer(self, state, action, reward, state_prime, terminal):
        self.buffer.append((state, action, reward, state_prime, terminal))

    def update_network(self):

        samples = self.sample_buffer()
        if len(samples) < 2:
            return
        states_init, actions_init, rewards_init, states_primes_init, terminals_init = zip(*samples)

        states = np.stack(states_init, axis=0)
        states = torch.from_numpy(states).float().to(self.device) / 255.0

        states_primes = np.stack(states_primes_init, axis=0)
        states_primes = torch.from_numpy(states_primes).float().to(self.device) / 255.0

        actions = torch.tensor(actions_init).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards_init).unsqueeze(1).to(self.device)
        terminals = torch.tensor(terminals_init, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_max, _ = torch.max(self.network(states_primes), dim=1)
        q_max_ = q_max.unsqueeze(1).to(self.device)

        td = rewards + self.gamma * q_max * (1 - terminals)

        q_of_actions_taken = self.target_network(states).gather(1, actions)
        error = self.loss(q_of_actions_taken, td)
        self.optimizer.zero_grad()
        error.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def sample_buffer(self):
        max_size = min(len(self.buffer), self.batch_size)
        return random.sample(self.buffer, max_size)

    def save_model(self):
        torch.save(self.network.state_dict(), 'models/DDQN_model.pt')
    def load_model(self, model_name=None):
        if model_name is None:
            raise ValueError("Model name cannot be None")
        self.network.load_state_dict(torch.load(f'models/{model_name}.pt'))

