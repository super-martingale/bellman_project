import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd
from wgan_gp.const import EPSILON


class Critic_1D(nn.Module):
    def __init__(self, input_size, output_size):
        # configurations
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        channels_l1 = 32
        channels_l2 = 32

        self.layer1 = nn.Linear(self.input_size, channels_l1)
        self.layer2 = nn.Linear(channels_l1, channels_l2)
        self.layer3 = nn.Linear(channels_l2, self.output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return F.sigmoid(x) #TODO: check net structure with Dror



class Generator_1D(nn.Module):
    def __init__(self, z_size, action_size, state_size, output_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.action_size = action_size
        self.state_size = state_size
        self.output_size = output_size

        channels_l1 = 32
        channels_l2 = 32

        self.layer1 = nn.Linear(self.input_size, channels_l1)
        self.layer2 = nn.Linear(channels_l1, channels_l2)
        self.layer3 = nn.Linear(channels_l2, self.output_size)


    def forward(self, z, s, a):
        x = torch.stack([z,s,a])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return F.softplus(x) #TODO: check net structure with Dror


class WGAN_1D(nn.Module):
    def __init__(self, replay_buffer, batch_size=32, z_size=1):
        # configurations
        super().__init__()
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.z_size = z_size

        # components
        self.critic = Critic_1D(
            input_size=self.input_size,
            output_size=self.output_size
        )
        self.generator = Generator_1D(
            z_size=self.z_size,
            action_size=self.action_size,
            state_size=self.state_size,
            output_size=self.output_size
        )

    @property
    def name(self):
        return (
            'WGAN_1D-GP'
            '-z{z_size}'
        ).format(
            z_size=self.z_size,
        )

    def c_loss(self, x, z, return_g=False):
        g = self.generator(z)
        c_x = self.critic(x).mean()
        c_g = self.critic(g).mean()
        l = -(c_x-c_g)
        return (l, g) if return_g else l

    def g_loss(self, z, return_g=False):
        g = self.generator(z)
        l = -self.critic(g).mean()
        return (l, g) if return_g else l

    def sample_image(self, size):
        return self.generator(self.sample_noise(size))

    def sample_noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * .1
        return z.cuda() if self._is_on_cuda() else z

    def gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a.cuda() if self._is_on_cuda() else a
        a = a\
            .expand(x.size(0), x.nelement()//x.size(0))\
            .contiguous()\
            .view(
                x.size(0),
                self.input_size,
                self.action_size,
                self.action_size
            )
        interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
        c = self.critic(interpolated)
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
