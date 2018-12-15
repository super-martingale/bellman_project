import torch
from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np, math
import os

from policy_tools.replay_buffer import ReplayBuffer
from policy_tools.DQN_1D import DQN_1D
from environments.env_factory import EnvFactory

class TrainPolicy():
    def __init__(self, env, model, device, load_pretrained_model=True, SAVE_DIR=None, use_ipython=False):
        self.use_ipython = use_ipython

        self.env = env
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters())

        self.iterations_per_epoch = 100000
        self.num_save_every_epochs = 10

        if SAVE_DIR is None:
            self.SAVE_DIR = 'saved_data'
        else:
            self.SAVE_DIR = SAVE_DIR

        self.MODEL_DIR = os.path.join(self.SAVE_DIR, 'models')
        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
        if not os.path.exists(os.path.join(self.SAVE_DIR, 'figures')):
            os.makedirs(os.path.join(self.SAVE_DIR, 'figures'))

        self.SAVE_MODEL_PATH = os.path.join(self.MODEL_DIR, self.model.name)
        self.epoch = 1
        if load_pretrained_model and os.path.exists(self.SAVE_MODEL_PATH):
            self.epoch, self.loss = self.load_model()


        self.gamma = 0.99
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 10000
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.init_pull_from_replay = 10000
        replay_capacity = 30000

        self.replay_buffer = ReplayBuffer(replay_capacity)

    def train(self):
        num_iterations = 1400000
        batch_size = 32
        game_number = 0
        steps_in_game = 0
        losses = []
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        for iter in range(1, num_iterations + 1):
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
                print('done game #{}. Number of steps in game {}'.format(game_number, steps_in_game))
                game_number += 1
                episode_reward = 0
                steps_in_game = 0

            if len(self.replay_buffer) > self.init_pull_from_replay:
                loss = self.compute_td_loss(batch_size)
                losses.append(loss.data[0])

            if iter % self.iterations_per_epoch == 0:
                mean_loss = torch.mean(torch.stack(losses))
                self.plot(iter, all_rewards, losses, mean_loss)
                self.epoch += 1

                all_rewards = []
                losses = []

            if steps_in_game % 1000 == 0:
                steps_in_game = 0
                state = self.env.reset()


            if iter % (self.iterations_per_epoch * self.num_save_every_epochs) == 0:
                self.save_model(self.epoch, mean_loss)

    def compute_td_loss(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

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
            plt.savefig(os.path.join(self.SAVE_DIR, 'figures', 'iter'+str(iter)+'__loss'+str(int(mean_loss.data.numpy()))))

    def save_model(self, epoch, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, self.SAVE_MODEL_PATH)

    def load_model(self):
        checkpoint = torch.load(self.SAVE_MODEL_PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        self.model.eval()
        return epoch, loss

    def plot_value_function(self):

        all_q_value = torch.tensor([])
        all_state = torch.tensor([])
        batch_size = 32

        for iter in range(1,1000):
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

            state = Variable(torch.FloatTensor(np.float32(state)))
            next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
            action = Variable(torch.LongTensor(action))
            reward = Variable(torch.FloatTensor(reward))
            done = Variable(torch.FloatTensor(done))

            q_values = self.model(state)
            next_q_values = self.model(next_state)

            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

            all_q_value = torch.cat([all_q_value, expected_q_value])
            all_state = torch.cat([all_state, state])

        plt.scatter(all_state.detach().numpy(), all_q_value.detach().numpy())
        plt.show()

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    environment_name = 'windshelter'
    env = EnvFactory.get_env(name=environment_name)

    model = DQN_1D(env)

    train_policy = TrainPolicy(env, model, device)
    train_policy.train()

