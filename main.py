import time

import gym
import numpy as np
import torch
from agents.DDQN import DDQNAgent
from utils.config import load_config
import random


def main():
    config = load_config()
    verbose = config.verbose
    start_time = time.time()

    epsilon = config.epsilon
    ep_decay = config.epsilon_decay
    epsilon_min = config.epsilon_min

    if config.seed is not None:  # sets seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    if config.render:
        env = gym.make(config.env_name, obs_type="ram", full_action_space=False, render_mode='human')  #render_mode='human'
    else:
        env = gym.make(config.env_name, obs_type="ram", full_action_space=False)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size, config.buffer_size,
                      config.learning_rate, config.NN_width,
                      config.gamma, config.batch_size)
    if config.train:
        step_counter = 0
        for episode in range(config.num_episodes):
            state, info = env.reset()
            terminal = False
            episode_reward = 0
            if step_counter % config.update_interval == 0:
                agent.update_target_network()
            while not terminal:
                action = agent.get_action(state, epsilon)
                state_prime, reward, terminal, _, info = env.step(action)
                agent.add_to_buffer(state, action, reward, state_prime, terminal)
                agent.update_network()
                state = state_prime
                episode_reward += reward
            step_counter += 1
            if epsilon > epsilon_min:
                epsilon *= ep_decay
            if verbose:
                print('Episode: ', step_counter, "|", "Total Run time (minutes): ", round((time.time() - start_time)/60, 2),
                      "|", "Episode Reward: ", episode_reward,
                      "|", "Current Epsilon: ", round(epsilon,3),)
            agent.save_model()

    if config.test:
        agent.load_model(model_name="DDQN_model")
        state, info = env.reset()
        terminal = False
        total_reward = 0
        while not terminal:
            action = agent.get_action(state, 0)
            state_prime, reward, terminal, _, info = env.step(action)
            state = state_prime
            total_reward += reward
        print('Total Reward: ', total_reward)






if __name__ == '__main__':
    main()
