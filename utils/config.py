class Config:
    def __init__(self):
        # Agent settings
        self.learning_rate = 1e-5
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Environment settings
        # list of games: ALE/Assault-v5,
        self.env_name = "SpaceInvaders-v0"

        # Training settings
        self.num_episodes = 1000
        self.max_steps_per_episode = 200
        self.save_interval = 100
        self.buffer_size = 10000
        self.update_interval = 50

        # Evaluation settings
        self.evaluate_every = 100
        self.num_evaluation_episodes = 10

        # Paths for saving and loading
        self.model_save_path = "models/"

        # Reproducibility
        self.seed = 0

        #Neural network
        self.NN_width=256

def load_config():
    return Config()
