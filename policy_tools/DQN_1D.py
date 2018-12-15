import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import random, math

from project_utils.gen_utils import GenUtils

device = GenUtils.get_device()


class DQN_1D(nn.Module):

    def __init__(self, env):
        super(DQN_1D, self).__init__()

        self.dtype = torch.float
        self.device = device

        self.input = torch.tensor(env.observation_space.shape[0])
        self.output = env.action_space.n

        channels_l1 = 32
        channels_l2 = 32

        self.layer1 = nn.Linear(self.input, channels_l1)
        self.layer2 = nn.Linear(channels_l1, channels_l2)
        self.layer3 = nn.Linear(channels_l2, self.output)
        #self.bn1 = nn.BatchNorm1d()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = x.view(x.size(0), -1) #TODO: check out 'view' function is necessery
        output = F.softplus(self.layer3(x))
        return output

    def act(self, state, epsilon):
        assert state.shape == torch.tensor([self.input]).shape, 'State is not of the correct shape'
        if random.random() > epsilon:
            state = Variable(torch.tensor(state, device=self.device, dtype=self.dtype).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].type(self.dtype) #.type(torch.FloatTensor).to(device)
        else:
            action = torch.tensor(random.randrange(self.output), device=self.device, dtype=self.dtype) #.type(torch.FloatTensor).to(device)
            #print(action)
        return action #.type(torch.FloatTensor).to(self.device)


    def print_model(self):
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        # # Print optimizer's state_dict
        # print("Optimizer's state_dict:")
        # for var_name in self.optimizer.state_dict():
        #     print(var_name, "\t", self.optimizer.state_dict()[var_name])

    @property
    def name(self):
        return (
            'DQN_1D'
            '-{states_size}'
            '-{action_size}'
        ).format(
            states_size=self.input, action_size=self.output
        )

