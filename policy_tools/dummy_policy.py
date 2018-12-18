import torch



class DummyPolicy():
    def act(self, state, epsilon):
        action = 1. -  state.abs()
        return action

    @property
    def name(self):
        return (
            'Dummy_policy'
        )
