import torch
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np, math
import os

from policy_tools.replay_buffer import ReplayBuffer
from policy_tools.DQN_1D import DQN_1D
from environments.env_factory import EnvFactory
from project_utils.gen_utils import GenUtils

from policy_tools.policy_functions import bellman_opt


device = GenUtils.get_device()
GenUtils.set_device_config()


class PolicyWrapper():
    '''
    PlicyWrapper receives a policy network and an gym environment.
    It supports actions such as train policy network, plot value function, create replay buffer, compute loss
    '''
    def __init__(self,
                 env,
                 model,
                 load_pretrained_model=True,
                 checkpoint='saved_data/checkpoints',
                 save_data_dir='saved_data',
                 use_ipython=False
                 ):
        self.use_ipython = use_ipython

        self.env = env
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters())

        self.iterations_per_epoch = 500
        self.num_save_every_epochs = 10

        self.save_data_dir = save_data_dir
        self.checkpoint = checkpoint
        assert checkpoint is not None, 'Please provide checkpoint directory'
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
        self.SAVE_MODEL_PATH = os.path.join(self.checkpoint, self.model.name)

        if not os.path.exists(os.path.join(self.save_data_dir, 'figures')):
            os.makedirs(os.path.join(self.save_data_dir, 'figures'))

        self.epoch = 1


        self.gamma = 0.95
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 10000
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        replay_capacity = 30000
        self.init_pull_from_replay = 10000  # Iterations after which to start training from replay buffer

        self.replay_buffer = ReplayBuffer(replay_capacity)


    def train_model(self, num_iterations =100000, resume_trainings=True):
        if resume_trainings and os.path.exists(self.SAVE_MODEL_PATH):
            self.epoch, self.loss = self.load_model()

        batch_size = 4096
        game_number = 0
        steps_in_game = 0
        losses = []
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        state.to(device)
        for iter in range(1, num_iterations + self.init_pull_from_replay):
            epsilon = self.epsilon_by_frame(iter)
            action = self.model.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            steps_in_game +=1
            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                game_number += 1
                episode_reward = 0
                steps_in_game = 0

            if len(self.replay_buffer) > self.init_pull_from_replay:
                loss = self.compute_td_loss(batch_size)
                losses.append(loss.data)

                if iter % self.iterations_per_epoch == 0:
                    print('Finished epoch #{} . Total Iteretion {}/{}'.format(self.epoch, iter -  self.init_pull_from_replay, num_iterations))
                    mean_loss = torch.mean(torch.stack(losses))
                    self.plot(iter, all_rewards, losses, mean_loss.to('cpu'))
                    self.epoch += 1

                    all_rewards = []
                    losses = []

                    if self.checkpoint is not None and self.epoch % self.num_save_every_epochs == 0:
                        self.save_model(self.epoch, mean_loss)

                if steps_in_game % 1000 == 0:
                    steps_in_game = 0
                    state = self.env.reset()

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done, next_action = self.replay_buffer.sample(batch_size)

        state = Variable(torch.tensor(np.float32(state)).to(device))
        next_state = Variable(torch.tensor(np.float32(next_state)).to(device))
        action = Variable(torch.LongTensor(action).to(device))
        reward = Variable(torch.tensor(reward).to(device))
        done = Variable(torch.tensor(done).to(device))

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = bellman_opt(reward.type(torch.float), self.gamma, next_q_value, (done).type(torch.float))

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


    def plot(self, iter, rewards, losses, mean_loss):
        if self.use_ipython:
            from IPython.display import clear_output
            clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (iter, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('Losses. Mean loss{}'.format(mean_loss))
        plt.plot(losses)
        if self.use_ipython:
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_data_dir, 'figures', 'iter'+str(iter)+'__loss'+str(int(mean_loss.data.numpy()))))

    def generate_replay_buffer(self, capacity):
        replay_buffer = ReplayBuffer(capacity)

        state = self.env.reset()
        state.to(device)

        while not replay_buffer.is_full():
            action = self.model.act(state, epsilon=0)
            next_state, reward, done, _ = self.env.step(action)
            next_action = self.model.act(next_state, epsilon=0)
            replay_buffer.push(state, action, reward, next_state, done, next_action)
            if done:
                state = self.env.reset()

        return replay_buffer


    def save_model(self, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.SAVE_MODEL_PATH)

    def load_model(self):
        assert os.path.exists(self.SAVE_MODEL_PATH), 'path {} does not contain DQN model'.format(self.SAVE_MODEL_PATH)
        checkpoint = torch.load(self.SAVE_MODEL_PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.model.eval()
        return epoch, loss


if __name__=="__main__":


    environment_name = 'windshelter'
    env = EnvFactory.get_env(name=environment_name)

    model = DQN_1D(input=env.state_dim, output=env.action_space.n)

    train_policy = PolicyWrapper(env, model)
    train_policy.train()

    train_policy.plot_value_function()

