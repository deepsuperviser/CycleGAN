import argparse
import os
from dog_solver import Solver
from torch.backends import cudnn
import torch
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(1)


def main(config):

    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    real_dog = torch.load('real_dog_data.pt').to(dtype=torch.float32)
    painting_dog = torch.load('painting_dog_data.pt').to(dtype=torch.float32)
    # print(real_dog.shape, torch.max(real_dog), torch.min(real_dog))
    # torch.Size([152, 3, 256, 256]) tensor(255.) tensor(0.)
    # print(painting_dog.shape, torch.max(painting_dog), torch.min(painting_dog))
    # torch.Size([133, 3, 256, 256]) tensor(255.) tensor(0.)
    real_dog = torch.div(torch.sub(torch.div(real_dog, 255.0), 0.5), 0.5)
    # print(real_dog.shape, torch.max(real_dog), torch.min(real_dog))
    # torch.Size([152, 3, 256, 256]) tensor(1.) tensor(-1.)
    painting_dog = torch.div(torch.sub(torch.div(painting_dog, 255.0), 0.5), 0.5)
    # print(painting_dog.shape, torch.max(painting_dog), torch.min(painting_dog))
    # torch.Size([133, 3, 256, 256]) tensor(1.) tensor(-1.)

    real_dog_set = TensorDataset(real_dog)
    painting_dog_set = TensorDataset(painting_dog)
    real_dog_loader = DataLoader(dataset=real_dog_set, shuffle=True, batch_size=config.batch_size)
    painting_dog_loader = DataLoader(dataset=painting_dog_set, shuffle=True, batch_size=config.batch_size)

    solver = Solver(config, svhn_loader=real_dog_loader, mnist_loader=painting_dog_loader)
    cudnn.benchmark = True
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--use_reconst_loss', type=bool, default=True)
    parser.add_argument('--use_labels', type=bool, default=False)
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument('--train_iters', type=int, default=40000)  # default=40000
    parser.add_argument('--batch_size', type=int, default=16)  # determine the figure
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--log_step', type=int, default=400)
    parser.add_argument('--sample_step', type=int, default=800)

    g_config = parser.parse_args()
    main(g_config)
