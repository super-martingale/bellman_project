import torch
from torch.autograd import Variable
import gym
from gym.utils import seeding
from gym import spaces

from env_utils.discrete_list import DiscreteValueList


class Windshelter(gym.Env):
    def __init__(self, A=1., B=0.3, D=0.3, threshold_shelter=0.5, reward_func=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.dtype = torch.float
        self.A = torch.tensor(A, device=self.device, dtype=self.dtype)
        self.B = torch.tensor(B, device=self.device, dtype=self.dtype)
        self.D = torch.tensor(D, device=self.device, dtype=self.dtype)
        self.threshold_shelter = torch.tensor(threshold_shelter, device=self.device, dtype=self.dtype)
        self.state = self.sample_init_state(state_size=1)
        if reward_func == None:
            self.get_reward = self._get_reward
        else:
            self.get_reward = reward_func

        self.game_endpoint = 1
        self.action_space = DiscreteValueList([torch.tensor(0., device=self.device, dtype=self.dtype),torch.tensor(1., device=self.device, dtype=self.dtype)])
        self.observation_space = spaces.Box(low=-2*self.game_endpoint, high= 2*self.game_endpoint, shape=(1,1), dtype=float)
        self._seed()



    def sample_init_state(self, state_size):
        z = torch.rand(state_size, device=self.device, dtype= self.dtype)
        return z

    def sample_env(self, s, a):
        assert self.action_space.contains(a), "%r (%s) invalid"%(a, type(a))
        a = torch.tensor(a)
        s = torch.tensor(s)
        D_val = Variable(self.D * (torch.abs(s) > self.threshold_shelter).type(torch.FloatTensor))
        w_D_val = D_val * torch.randn(1)
        next_state = self.A * s + self.B * a + w_D_val
        reward = self.get_reward(s, a)
        done = self._get_done(next_state)
        return next_state, reward, done

    def _get_reward(self, s, a, done):
        if done:
            reward = -10
        else:
            reward = 1
        return reward

    def _get_done(self, next_state):
        done = (abs(next_state) > self.game_endpoint)
        return done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        #a = torch.tensor([a])
        assert self.action_space.contains(a), "%r (%s) invalid"%(a, type(a))
        if a == 0:
            a = -1
        D_val = self.D * (torch.abs(self.state) > self.threshold_shelter).type(self.dtype)
        w_D_val = D_val * torch.randn(1)
        next_state = self.A * self.state + self.B * a + w_D_val
        self.state = next_state
        done = self._get_done(next_state)
        reward = self.get_reward(self.state, a, done)
        return next_state, reward, done, {}

    def reset(self):
        self.state = self.sample_init_state(state_size=1)
        return self.state

    def render(self, mode='human', close=False):
        raise NotImplemented


if __name__ == '__main__':
    w = Windshelter()
    w.sample_env(s=1., a = 1.)
    w.step(a = 1.)