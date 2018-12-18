import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_value_functions(policy_wrapper, gan_model):
    policy_model = policy_wrapper.model


    states = np.arange(-1., 1.0, 0.01)
    actions = [0,1]
    for action in actions:
        q_dist_queue_gan = []
        states_queue = []
        states_queue_dist = []
        q_values_queue_policy = []
        for state in states:
            #Save DQN Q-value:
            state = torch.tensor(state, dtype=torch.float).view(1,1)
            q_value = policy_model(state)
            q_value_action = q_value[0][action]
            q_values_queue_policy.append(q_value_action.unsqueeze(0))
            states_queue.append(state)
            #Save Bellman GAN Z-value:
            sample_gan = 20
            for sample in range(1,sample_gan):
                z = gan_model.sample_noise(size=1)
                dis_q = gan_model.sample_g(state.view(1,1), torch.tensor(action,dtype=torch.float).view(1,1), z)
                q_dist_queue_gan.append(dis_q)
                states_queue_dist.append(state)

        # Plot graphs:
        q_values_queue_policy = torch.cat(q_values_queue_policy)
        states_queue_dist = torch.cat(states_queue_dist)
        q_dist_queue_gan = torch.cat(q_dist_queue_gan)
        states_queue = torch.cat(states_queue)

        plt.figure(action)
        plt.title('Belman distribuiotnal reward with policy action={}'.format(action))
        plt.scatter(states_queue_dist.to('cpu').detach().numpy(), q_dist_queue_gan.to('cpu').detach().numpy(),
                    cmap='Purples')
        plt.savefig('saved_data/Belman distribuiotnal reward with policy action={}.png'.format(action))

        plt.figure(action+3)
        plt.title('DQN Q value for policy action={}'.format(action))
        plt.scatter(states_queue.to('cpu').detach().numpy(), q_values_queue_policy.to('cpu').detach().numpy(),cmap='viridis')
        plt.savefig('saved_data/DQN Q value for policy action={}.png'.format(action))


        plt.show()

def sample_q_dist(model):
    gan_model = model
    states = np.arange(-1., 1.0, 0.01)
    actions = [0,1]
    for action in actions:
        q_dist_queue_gan = []
        states_queue_dist = []
        for state in states:
            sample_gan = 20
            for sample in range(1,sample_gan):
                z = gan_model.sample_noise(size=1)
                dis_q = gan_model.sample_g(state.view(1,1), torch.tensor(action,dtype=torch.float).view(1,1), z)
                q_dist_queue_gan.append(dis_q)
                states_queue_dist.append(state)

        states_queue_dist = torch.cat(states_queue_dist)
        q_dist_queue_gan = torch.cat(q_dist_queue_gan)

        plt.figure(action)
        plt.title('Belman distribuiotnal reward with policy action={}'.format(action))
        plt.scatter(states_queue_dist.to('cpu').detach().numpy(), q_dist_queue_gan.to('cpu').detach().numpy(),
                    cmap='Purples')
        plt.savefig('saved_data/Belman distribuiotnal reward with policy action={}.png'.format(action))
