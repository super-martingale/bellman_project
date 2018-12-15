import torch
from torch.autograd import Variable
import torch.optim as optim

import os
import argparse

from policy_tools.DQN_1D import DQN_1D
from environments.env_factory import EnvFactory
from policy_tools.policy_wrapper import PolicyWrapper

from wgan_gp import utils
from project_utils.gen_utils import GenUtils


device = GenUtils.get_device()
GenUtils.set_device_config()


class BellmanGAN():
    def __init__(self, policy_net):
        self.policy_net = policy_net

    def train_policy(self):
        self.policy_wrapper = PolicyWrapper(env, policy_net)
        self.policy_wrapper.train()
        # policy_wrapper.plot_value_function()

    def load_policy(self):
        self.policy_wrapper = PolicyWrapper(env, policy_net, load_pretrained_model=True)

    def get_replay_buffer(self, capacity):
        self.replay_buffer = self.policy_wrapper.generate_replay_buffer(capacity)



if __name__=="__main__":


    parser = argparse.ArgumentParser('PyTorch Implementation of WGAN-GP')
    parser.add_argument('--z-size', type=int, default=100)
    parser.add_argument('--lamda', type=float, default=10.)

    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--weight-decay', type=float, default=1e-04)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-size', type=int, default=36)

    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--loss-log-interval', type=int, default=30)

    parser.add_argument('--environment_name', default='windshelter', type=str, choices=['windshelter','cartpole'])

    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda')

    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--test', action='store_true', dest='test')
    command.add_argument('--train', action='store_false', dest='test')

    args = parser.parse_args()


    env = EnvFactory.get_env(name=args.environment_name)
    policy_net = DQN_1D(env)
    bellman_gan = BellmanGAN(policy_net)


    bellman_gan.load_policy()
    bellman_gan.get_replay_buffer(capacity=1000)


    utils.gaussian_intiailize(bellman_gan, 0.02)


    if GenUtils.is_cuda():
        bellman_gan.cuda()

    if args.test:
        path = os.path.join(args.sample_dir, '{}-sample'.format(bellman_gan.name))
        utils.load_checkpoint(bellman_gan, args.checkpoint_dir)
        utils.test_model(bellman_gan, args.sample_size, path)
    else:
        train_bellmanGAN(
            bellman_gan, bellman_gan.replay_buffer,
            collate_fn=dataset_config.get('collate_fn', None),
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            lamda=args.lamda,  # TODO: change to 0.1
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            epochs=args.epochs,
            d_trains_per_g_train=args.d_trains_per_g_train,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            image_log_interval=args.image_log_interval,
            loss_log_interval=args.loss_log_interval,
            resume=args.resume,
            cuda=cuda,
        )

