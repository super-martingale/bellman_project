import torch
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np, math
import os

from policy_tools.replay_buffer import ReplayBuffer
from policy_tools.DQN_1D import DQN_1D
from environments.env_factory import EnvFactory
from policy_tools.policy_wrapper import PolicyWrapper

from project_utils.gen_utils import GenUtils


device = GenUtils.get_device()
GenUtils.set_device_config()


class BellmanGAN():


    def train_policy(self):
        self.policy_wrapper = PolicyWrapper(env, policy_net)
        self.policy_wrapper.train()
        # policy_wrapper.plot_value_function()

    def load_policy(self):
        self.policy_wrapper = PolicyWrapper(env, policy_net, load_pretrained_model=True)

    def get_replay_buffer(self, capacity):
        self.policy_wrapper.generate_replay_buffer(capacity)

    def train_critic(self):
        pass

    def train_generator(self):
        pass


if __name__=="__main":



    environment_name = 'windshelter'
    env = EnvFactory.get_env(name=environment_name)

    policy_net = DQN_1D(env)

    bellman_gan = BellmanGAN()
    #bellman_gan.train_policy()
    bellman_gan.load_policy()
    bellman_gan.get_replay_buffer(capacity=100000)

    bellman_gan.train_critic()
    bellman_gan.train_generator()