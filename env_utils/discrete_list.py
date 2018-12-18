import numpy as np
import gym

class DiscreteValueList(gym.Space):
    """
    {[values]}

    Example usage:
    self.observation_space = spaces.Discrete(DiscreteBinary)
    """
    def __init__(self, values):
        self.values = values
        gym.Space.__init__(self, (), np.int64)
        self.n = len(self.values)
        self.type = type(self.values[0])

    def sample(self):
        return self.values[gym.spaces.np_random.randint(len(self.values))]

    def contains(self, x):
        # if not isinstance(x, self.type):
        #     raise AttributeError('input type'+str(type(x))+'does not match type of defined actions: '+str(self.type))
        # if isinstance(x,int) or isinstance(x, float):
        #     as_int =
        # elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
        #     as_int = float(x)
        # else:
        #     return False
        return x in self.values

    def __repr__(self):
        return "DiscreteList" + str(self.values)

    def __eq__(self, other):
        return self.values == other.values
