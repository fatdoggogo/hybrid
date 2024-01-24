import gym


class Car2DEnv(gym.Env):
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        self.state = None

        def step(self, action):

            return self.state, reward, done, {}

        def reset(self):
            return self.state

        def render(self, mode='human'):
            return None

        def close(self):
            return None

