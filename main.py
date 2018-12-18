import torch
from torch.autograd import Variable
import torch.optim as optim

import os
import argparse

from policy_tools.DQN_1D import DQN_1D
from environments.env_factory import EnvFactory
from policy_tools.policy_wrapper import PolicyWrapper
from wgan_gp.model_1D import WGAN_1D
from wgan_gp import wg_utils
from project_utils.gen_utils import GenUtils
from train_bellmanGAN import train_bellmanGAN
from env_utils import test_utils

device = GenUtils.get_device()
GenUtils.set_device_config()


class BellmanGAN():

    def get_replay_buffer(self, capacity):
        self.replay_buffer = self.policy_wrapper.generate_replay_buffer(capacity)



if __name__=="__main__":
    parser = argparse.ArgumentParser('PyTorch Implementation of WGAN-GP')
    parser.add_argument('--z-size', type=int, default=1)
    parser.add_argument('--lamda', type=float, default=10.)

    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--weight-decay', type=float, default=1e-04)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--d-trains-per-g-train', type=int, default=5)
    parser.add_argument('--sample-size', type=int, default=36)

    parser.add_argument('--environment_name', default='windshelter', type=str, choices=['windshelter','cartpole'])

    parser.add_argument('--resume-policy',action='store_false', help='if a policy model exists, load it from memory')
    parser.add_argument('--skip-policy-training', action='store_true',
                        help='policy will not be trained. valid only if a policy model is saved in memory and load-pretrained-policy=True')
    parser.add_argument('--resume-gan', action='store_false')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda')


    parser.add_argument('--checkpoint-dir', type=str, default='saved_data/checkpoints')
    parser.add_argument('--save-data-dir', type=str, default='saved_data')
    parser.add_argument('--loss-log-interval', type=int, default=30)
    parser.add_argument('--image-log-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=1000)


    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--test', action='store_true', dest='test')
    command.add_argument('--train', action='store_false', dest='test')
    args = parser.parse_args()


    env = EnvFactory.get_env(name=args.environment_name)
    policy_net = DQN_1D(env)
    policy_wrapper = PolicyWrapper(env=env,
                                   model=policy_net,
                                   checkpoint=args.checkpoint_dir,
                                   save_data_dir=args.save_data_dir,
                                   load_pretrained_model=args.resume_policy
                                   )

    if not args.skip_policy_training: # continue to train policy: (Training DQN module)
        policy_wrapper.train_model(num_iterations =20000, resume_trainings=True)
    else:
        policy_wrapper.load_model()
    # policy_wrapper.plot_value_function()
    print('Finished policy loading: {}'.format(policy_net.name))

    if not args.test:
        #Generate replay buffer according to policy:
        capacity = 10000
        replay_buffer = policy_wrapper.generate_replay_buffer(capacity)
        print('Finished generating replay buffer with {} samples'.format(capacity))

    #Load WGAN-GP model:
    gan = WGAN_1D(generator_output_size=1,
                  critic_output_size=1,
                  state_dim=env.state_dim, # observation_space.shape[0],
                  actions_option_num=env.actions_option_num,
                  z_size = 1)
    if GenUtils.is_cuda():
        gan.cuda()

    wg_utils.gaussian_intiailize(gan, 0.02)

    if args.test:
        #path = os.path.join(args.sample_dir, '{}-sample'.format(gan.name))
        wg_utils.load_checkpoint(gan, args.checkpoint_dir)
        print('Bellman GAN model loaded for testing')
        #wg_utils.test_model(gan, args.sample_size, path)
    else: # train
        train_bellmanGAN(
            gan, replay_buffer,
            collate_fn=None,#dataset_config.get('collate_fn', None),
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
            resume=args.resume_gan,
            cuda= GenUtils.is_cuda()
        )
        print('Finished training Bellman GAN')

    print('Program finished')

    test_utils.plot_value_functions(policy_wrapper, gan)


