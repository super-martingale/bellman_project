

def bellman_opt(reward, gamma, next_value, done):
    expected_value = reward*(1-gamma) + gamma * next_value * (1. - done)
    return expected_value