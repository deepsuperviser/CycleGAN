import torch
import torch.nn as nn
import os
import scipy.io
import numpy as np
from torch import optim
from dog_model import G12, G21, D1, D2
from torch.autograd import Variable
from PIL import Image


class Solver(object):
    def __init__(self, config, svhn_loader, mnist_loader):
        self.device = torch.device("cuda:2")
        self.svhn_loader = svhn_loader
        self.mnist_loader = mnist_loader
        self.use_labels = config.use_labels
        self.g12 = G12(conv_dim=config.g_conv_dim)
        self.g21 = G21(conv_dim=config.g_conv_dim)
        self.d1 = D1(conv_dim=config.d_conv_dim, use_labels=self.use_labels)
        self.d2 = D2(conv_dim=config.d_conv_dim, use_labels=self.use_labels)
        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())
        self.g_optimizer = optim.Adam(g_params, config.lr, (config.beta1, config.beta2))
        self.d_optimizer = optim.Adam(d_params, config.lr, (config.beta1, config.beta2))

        self.use_reconst_loss = config.use_reconst_loss
        self.num_classes = config.num_classes
        self.train_iters = config.train_iters
        self.batch_size = config.batch_size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.sample_path = config.sample_path
        
        if torch.cuda.is_available():
            self.g12.to(self.device)
            self.g21.to(self.device)
            self.d1.to(self.device)
            self.d2.to(self.device)
        

    def merge_images(self, sources, targets):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row*h, row*w*2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
            merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
        return merged.transpose(1, 2, 0)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.to(self.device)
        return Variable(x)

    def to_data(self, x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        
        svhn_iter = iter(self.svhn_loader)
        mnist_iter = iter(self.mnist_loader)
        # print(len(svhn_iter), len(mnist_iter)) 152/16=10  133/16=9
        iter_per_epoch = min(len(svhn_iter), len(mnist_iter)) - 1
        # print(svhn_iter.next()[0].shape) torch.Size([64, 3, 32, 32])
        # print(svhn_iter.next()[0].shape) torch.Size([16, 3, 256, 256])
        
        fixed_svhn = self.to_var(svhn_iter.next()[0])
        # print(mnist_iter.next()[0].shape) torch.Size([64, 1, 32, 32])
        # print(mnist_iter.next()[0].shape) torch.Size([16, 3, 256, 256])
        fixed_mnist = self.to_var(mnist_iter.next()[0])

        for step in range(self.train_iters + 1):
            if (step+1) % iter_per_epoch == 0:
                mnist_iter = iter(self.mnist_loader)
                svhn_iter = iter(self.svhn_loader)
            svhn = svhn_iter.next()[0]
            svhn = self.to_var(svhn)
            mnist = mnist_iter.next()[0]
            mnist = self.to_var(mnist)
            # train D
            self.reset_grad()
            out = self.d1(mnist)
            d1_loss = torch.mean((out - 1) ** 2)
            out = self.d2(svhn)
            d2_loss = torch.mean((out - 1) ** 2)
            d_real_loss = d1_loss + d2_loss
            d_real_loss.backward()
            self.d_optimizer.step()

            # train with fake images
            self.reset_grad()
            fake_svhn = self.g12(mnist)
            out = self.d2(fake_svhn)
            d2_loss = torch.mean(out ** 2)

            fake_mnist = self.g21(svhn)
            out = self.d1(fake_mnist)
            d1_loss = torch.mean(out ** 2)

            d_fake_loss = d1_loss + d2_loss
            d_fake_loss.backward()
            self.d_optimizer.step()

            # train G
            self.reset_grad()
            fake_svhn = self.g12(mnist)
            out = self.d2(fake_svhn)
            reconst_mnist = self.g21(fake_svhn)
            g_loss = torch.mean((out - 1) ** 2)
            if self.use_reconst_loss:
                g_loss += torch.mean((mnist - reconst_mnist) ** 2)
            g_loss.backward()
            self.g_optimizer.step()

            self.reset_grad()
            fake_mnist = self.g21(svhn)
            out = self.d1(fake_mnist)
            reconst_svhn = self.g12(fake_mnist)
            g_loss = torch.mean((out - 1) ** 2)
            if self.use_reconst_loss:
                g_loss += torch.mean((svhn - reconst_svhn) ** 2)
            g_loss.backward()
            self.g_optimizer.step()
            if (step + 1) % self.sample_step == 0:  # self.sample_step
                fake_svhn = self.g12(fixed_mnist)
                fake_mnist = self.g21(fixed_svhn)
                mnist, fake_mnist = self.to_data(fixed_mnist), self.to_data(fake_mnist)
                svhn, fake_svhn = self.to_data(fixed_svhn), self.to_data(fake_svhn)
                with torch.no_grad():
                    fake_svhn = (fake_svhn * 0.5 + 0.5) * 255.0
                    svhn = (svhn * 0.5 + 0.5) * 255.0
                    fake_mnist = (fake_mnist * 0.5 + 0.5) * 255.0
                    mnist = (mnist * 0.5 + 0.5) * 255.0
                merged = self.merge_images(mnist, fake_svhn)
                path = os.path.join(self.sample_path, 'sample-%d-m-s.png' % (step + 1))
                # print(type(merged)) <class 'numpy.ndarray'>
                # print(merged.shape)  (256, 512, 3)
                im = Image.fromarray(np.uint8(merged))
                im.save(path)
                # scipy.misc.imsave(path, merged)
                print('saved %s' % path)
                merged = self.merge_images(svhn, fake_mnist)
                path = os.path.join(self.sample_path, 'sample-%d-s-m.png' % (step + 1))
                im = Image.fromarray(np.uint8(merged))
                im.save(path)
                # scipy.misc.imsave(path, merged)
                print('saved %s' % path)
