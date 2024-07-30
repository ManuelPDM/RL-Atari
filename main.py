import gym
import numpy as np
import torch
from agents.DDQN import DDQNAgent
from utils.config import load_config
import random


def main():
    config = load_config()
    if config.seed is not None:  # sets seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    env = gym.make(config.env_name, obs_type="ram", full_action_space=False, )  #render_mode='human'
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size, config.buffer_size,
                      config.epsilon, config.epsilon_decay, config.learning_rate, config.NN_width,config.epsilon_min)
    step_counter = 0
    for episode in range(config.num_episodes):
        state, info = env.reset()
        terminal = False
        if step_counter % config.update_interval == 0:
            agent.update_target_network()
        while not terminal:
            action = agent.get_action(state)
            state_prime, reward, terminal, _, info = env.step(action)
            agent.add_to_buffer(state, action, reward, state_prime, terminal)
            agent.update_network()
        step_counter += 1



if __name__ == '__main__':
    main()
