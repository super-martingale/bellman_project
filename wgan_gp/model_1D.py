import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd
from wgan_gp.const import EPSILON

from policy_tools.policy_functions import bellman_opt


class Critic_1D(nn.Module):
    def __init__(self, input_size, actions_option_num, state_dim, output_size):
        # configurations
        super().__init__()
        self.input_size = input_size
        self.actions_option_num = actions_option_num
        self.state_dim = state_dim
        self.output_size = output_size

        net_input = self.input_size +self.actions_option_num + self.state_dim
        channels_l1 = 32
        channels_l2 = 32

        self.layer1 = nn.Linear(net_input, channels_l1, bias=True)
        self.layer2 = nn.Linear(channels_l1, channels_l2, bias=True)
        self.layer3 = nn.Linear(channels_l2, self.output_size, bias=False)

    def forward(self, x, s, a):
        a_one_hot = (a == torch.arange(self.actions_option_num).reshape(1, self.actions_option_num).float()).float()
        x = torch.cat([x, s, a_one_hot],dim=1)
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = self.layer3(x)
        return x #F.sigmoid(x) #TODO: check net structure with Dror



class Generator_1D(nn.Module):
    def __init__(self, z_size, actions_option_num, state_dim, output_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.actions_option_num = actions_option_num
        self.state_dim = state_dim
        self.output_size = output_size

        channels_l1 = 32
        channels_l2 = 32

        self.input_size = self.z_size+ self.actions_option_num+ self.state_dim
        self.layer1 = nn.Linear(self.input_size, channels_l1)
        self.layer2 = nn.Linear(channels_l1, channels_l2)
        self.layer3 = nn.Linear(channels_l2, self.output_size)


    def forward(self, z, s, a):
        a_one_hot = (a == torch.arange(self.actions_option_num).reshape(1, self.actions_option_num).float()).float()
        x = torch.cat([z,s, a_one_hot],dim=1)
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return x # F.softplus(x)


class WGAN_1D(nn.Module):
    def __init__(self, generator_output_size, critic_output_size, state_dim, actions_option_num, z_size=1):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.generator_output_size = generator_output_size
        self.critic_output_size = critic_output_size
        self.actions_option_num = actions_option_num
        self.state_dim = state_dim

        # components
        self.critic = Critic_1D(
            input_size=self.generator_output_size,
            actions_option_num=self.actions_option_num,
            state_dim=self.state_dim,
            output_size=self.critic_output_size
        )
        self.generator = Generator_1D(
            z_size=self.z_size,
            actions_option_num=self.actions_option_num,
            state_dim=self.state_dim,
            output_size=self.generator_output_size
        )

    @property
    def name(self):
        return (
            'WGAN_1D-GP'
            '-z{z_size}'
        ).format(
            z_size=self.z_size,
        )

    def c_loss(self, state, action, z, next_state, next_action, zz, reward, gamma, done, return_g=False):
        g = self.generator(z, s=state, a=action)
        g_real = self.generator(zz, s=next_state, a=next_action)
        c_real_input = bellman_opt(reward.view(g_real.shape), gamma, g_real, torch.tensor(done, dtype=torch.float))
        c_x = self.critic(c_real_input, state, action).mean()
        c_g = self.critic(g, state, action).mean()
        l = c_g-c_x
        return (l, g, g_real) if return_g else l

    def g_loss(self, state, action, z, next_state, next_action, zz, reward, gamma, done, return_g=False):
        g = self.generator(z, s=state, a=action)
        g_real = self.generator(zz, s=next_state, a=next_action)
        c_real_input = bellman_opt(reward.view(g_real.shape), gamma, g_real, torch.tensor(done, dtype=torch.float))
        c_g = self.critic(g, state, action).mean()
        c_x = self.critic(c_real_input, state, action).mean()
        l = -(c_g-c_x)
        return (l, g, g_real) if return_g else l

    def sample_noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * .1
        return z.cuda() if self._is_on_cuda() else z

    def gradient_penalty(self, x, g, state, action, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a.cuda() if self._is_on_cuda() else a
        # a = a\
        #     .expand(x.size(0), x.nelement()//x.size(0))\
        #     .contiguous()\
        #     .view(
        #         x.size(0),
        #         self.generator_output_size,
        #         self.state_dim,
        #         self.state_dim
        #     )
        interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
        c = self.critic(interpolated, state, action)
        gradients = autograd.grad(
            c, interpolated, grad_outputs=(
                torch.ones(c.size()).cuda() if self._is_on_cuda() else
                torch.ones(c.size())
            ),
            create_graph=True,
            retain_graph=True,
        )[0]
        return lamda * ((1-(gradients+EPSILON).norm(2, dim=1))**2).mean()

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def sample_g(self, state, action, z):
        g = self.generator(z, s=state, a=action)
        return g