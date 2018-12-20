import torch
import torch.nn.functional as F

from project_utils.gen_utils import GenUtils
device = GenUtils.get_device()


class DummyPolicy(torch.nn.Module):
    def __init__(self, input, output):
        super(DummyPolicy, self).__init__()

        self.input = input #torch.tensor(env.observation_space.shape[0])
        self.output = output
        self.device = device
        self.dtype = torch.float

        channels_l1 = 32

        self.layer1 = torch.nn.Linear(self.input, channels_l1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return x

    def act(self, state, epsilon):
        if state > 0:
            action = torch.tensor([1], device=self.device, dtype=self.dtype)
        else:
            action = torch.tensor([0], device=self.device, dtype=self.dtype)
        return action

    @property
    def name(self):
        return (
            'Dummy_policy'
        )
