import gym

class EnvFactory():
    @staticmethod
    def get_env(name):
        if name == 'cartpole':
            env_id = "CartPole-v0"
            env = gym.make(env_id)
        elif name == 'windshelter':
            env_id = "Windshelter-v0"
            env = gym.make(env_id)
        else:
            assert ModuleNotFoundError(name +' is not set s environment yet. Please add it')
        return env

if __name__=="__main__":
    env_fac = EnvFactory()
    env_fac.get_env(name='windshelter')