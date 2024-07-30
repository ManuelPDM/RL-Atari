from collections import deque

import numpy as np


class DDQNAgent:
    def __init__(self, state_size, action_size, buffer_size, epsilon, epsilon_decay, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

    def get_action(self, state):
        action = np.random.choice(self.action_size)
        return action

    def add_to_buffer(self, state, action, reward, state_prime, terminal):
        self.buffer.append((state, action, reward, state_prime, terminal))
    def update_network(self):
        pass
