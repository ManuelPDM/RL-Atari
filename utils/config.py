class Config:
    def __init__(self):
        # Main file settings to train and or test
        self.train = True
        self.test = False
        if self.test:
            self.render = True
        else:
            self.render = False

        # Agent settings
        self.learning_rate = 1e-4
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Environment settings
        # list of games: ALE/Assault-v5,
        self.env_name = "SpaceInvaders-v4"

        # Training settings
        self.num_episodes = 200
        self.max_steps_per_episode = 200
        self.save_interval = 100
        self.buffer_size = 10000
        self.batch_size = 500
        self.update_interval = 15

        # Reproducibility
        self.seed = 0

        #Neural network
        self.NN_width=256

        self.verbose = True

def load_config():
    return Config()
