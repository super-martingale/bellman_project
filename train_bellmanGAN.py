import torch
from tqdm import tqdm
from wgan_gp import wg_utils as utils
from env_utils.test_utils import plot_dist_reward_function

#from wgan_gp import visual
#from torchvision.utils import save_image
import os
def train_bellmanGAN(model, dataset, collate_fn=None,
          lr=1e-04, weight_decay=1e-04, beta1=0.5, beta2=.999, lamda=0.1,
          batch_size=64, sample_size=64, epochs=10,
          d_trains_per_g_train=2,
          checkpoint_dir='saved_data/BellmanWGAN_checkpoints',
          checkpoint_interval=1000,
          plot_iterval = 1000,
          image_log_interval=10,
          loss_log_interval=30,
          resume=False, cuda=False):
    # define the optimizers.
    generator_optimizer = torch.optim.Adam(
        model.generator.parameters(), lr=lr, betas=(beta1, beta2),
        weight_decay=weight_decay
    )
    critic_optimizer = torch.optim.Adam(
        model.critic.parameters(), lr=lr, betas=(beta1, beta2),
        weight_decay=weight_decay
    )

    # prepare the model and statistics.
    model.train()
    epoch_start = 1

    c_loss_history = []
    iteration_history = []
    # load checkpoint if needed.
    if resume and os.path.exists(os.path.join(checkpoint_dir, model.name)):
        iteration, epoch_start = utils.load_checkpoint(model, checkpoint_dir)
        #epoch_start = iteration // (len(dataset) // batch_size) + 1

    for epoch in range(epoch_start, epochs+1):
        data_loader = utils.get_data_loader(
            dataset, batch_size,
            cuda=cuda, collate_fn=collate_fn,
        )
        data_stream = tqdm(enumerate(data_loader, 1))
        for batch_index, data in data_stream:
            # unpack the data if needed.
            try:
                x, _ = data
            except ValueError:
                x = data
            # where are we?
            dataset_size = len(data_loader.dataset)
            dataset_batches = len(data_loader)
            iteration = (
                (epoch-1)*(dataset_size // batch_size) +
                batch_index + 1
            )

            state, action, reward, next_state, done, next_action = x
            if cuda: #TODO: check if you can move the entire replay buffer to GPU
                state = state.to('cuda')
                action =action.to('cuda')
                next_state = next_state.to('cuda')
                next_action = next_action.to('cuda')
                reward = torch.tensor(reward, dtype=torch.float)

            d_trains = (
                5 if (batch_index < 25 or batch_index % 500 == 0) else
                d_trains_per_g_train
            )
            gamma = 0.95
            # run the critic and backpropagate the errors.
            for _ in range(d_trains):
                critic_optimizer.zero_grad()
                z = model.sample_noise(batch_size)
                zz = model.sample_noise(batch_size)
                c_loss, g, g_real = model.c_loss(state, action, z, next_state, next_action, zz, reward, gamma, done, return_g=True)
                c_loss_gp = c_loss + model.gradient_penalty(g_real, g, state, action, lamda=lamda)
                c_loss_gp.backward()
                critic_optimizer.step()

            # run the generator and backpropagate the errors.
            generator_optimizer.zero_grad()
            z = model.sample_noise(batch_size)
            zz = model.sample_noise(batch_size)
            g_loss= model.g_loss(state, action, z, next_state, next_action, zz, reward, gamma, done, return_g=False)
            g_loss.backward()
            generator_optimizer.step()

            # update the progress.
            data_stream.set_description((
                'epoch: {epoch}/{epochs} | '
                'iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'g: {g_loss:.4} / '
                'w: {w_dist:.4}'
            ).format(
                epoch=epoch,
                epochs=epochs,
                iteration=iteration,
                trained=batch_index*batch_size,
                total=dataset_size,
                progress=(100.*batch_index/dataset_batches),
                g_loss=g_loss.data,
                w_dist=-c_loss.data,
            ))

            c_loss_history.append(c_loss)
            iteration_history.append(iteration)

            # send losses to the visdom server.
            # if iteration % loss_log_interval == 0:
            #     visual.visualize_scalar(
            #         -c_loss.data,
            #         'estimated wasserstein distance between x and g',
            #         iteration=iteration,
            #         env=model.name
            #     )
            #     visual.visualize_scalar(
            #         g_loss.data,
            #         'generator loss',
            #         iteration=iteration,
            #         env=model.name
            #     )

            # send sample images to the visdom server.
            # if iteration % image_log_interval == 0:
            #     visual.visualize_images(
            #         model.sample_image(sample_size).data,
            #         'generated samples',
            #         env=model.name
            #     )
                #save_image(model.sample_image(size=1).data[0], 'WGAN_generated_image.png')

            # save the model at checkpoints.
            if iteration % checkpoint_interval == 0:
                # notify that we've reached to a new checkpoint.
                print()
                print()
                print('#############')
                print('# checkpoint!')
                print('#############')
                print()

                utils.save_checkpoint(model, checkpoint_dir, iteration, epoch)

                print()
            if iteration % plot_iterval == 0:
                plot_dist_reward_function(model, epoch)