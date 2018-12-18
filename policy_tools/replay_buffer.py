import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import deque

class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done, next_action=None):
        state = state.to('cpu')
        action = action.to('cpu')
        #reward = reward.to('cpu')
        next_state = next_state.to('cpu')
        #state = np.expand_dims(state, 0)
        #next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done, next_action))

    def sample(self, batch_size):
        state, action, reward, next_state, done, next_action = zip(*random.sample(self.buffer, batch_size))
        return torch.cat(state).view(batch_size,-1), \
               action, \
              reward, \
                torch.cat(next_state).view(batch_size, -1), \
               done, \
               next_action

    def is_full(self):
        return len(self.buffer) == self.capacity

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]