"""
    Wasserstein Autoencoder with GAN critic.

    Theory by Tolstikhin et al., 2018. ()
    Original PyTorch implementation by schelotto (https://github.com/schelotto/Wasserstein_Autoencoders)
    Modified by Wilbert Copeland
"""

from pathlib import Path
import sys
import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR


ROOT_DIR = Path.cwd()

sys.path.append(str(ROOT_DIR))
from config import Config
from model import Encoder, Decoder, Critic


""" CONFIGURATION """

torch.manual_seed(2019)

class CancerConfig(Config):
    NAME = 'cancer'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


""" USER DEFINITIONS """

# parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')
# parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
# parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train (default: 100)')
# parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
# parser.add_argument('--dim_h', type=int, default=128, help='hidden dimension (default: 128)')
# parser.add_argument('--n_z', type=int, default=8, help='hidden dimension of z (default: 8)')
# parser.add_argument('--LAMBDA', type=float, default=10, help='regularization coef MMD term (default: 10)')
# parser.add_argument('--n_channel', type=int, default=1, help='input channels (default: 1)')
# parser.add_argument('--sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
# # args = parser.parse_args()
# args = parser.parse_args('')


""" """

# Display configuration settings
config = CancerConfig()
config.display()

# Prepare train and test data
train_dataset = MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
train_generator = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

test_dataset = MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
test_generator = DataLoader(dataset=test_dataset, batch_size=104, shuffle=False)

# Initiate network models
encoder = Encoder(config).to(config.DEVICE)
decoder = Decoder(config).to(config.DEVICE)
critic = Critic(config).to(config.DEVICE)

# Define reconstruction loss function
criterion = nn.MSELoss()

# Set all models to train mode
encoder.train()
decoder.train()
critic.train()

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr=config.LEARNING_RATE)
dec_optim = optim.Adam(decoder.parameters(), lr=config.LEARNING_RATE)
dis_optim = optim.Adam(critic.parameters(), lr=0.5 * config.LEARNING_RATE)

enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)

#

one = torch.Tensor([1]).to(config.DEVICE)
mone = one * -1
assert one.device == mone.device


""" Start training """

for epoch in range(config.NUM_EPOCHS):
    step = 0

    for images, _ in tqdm(train_generator):

        if torch.cuda.is_available():
            images = images.cuda()

        encoder.zero_grad()
        decoder.zero_grad()
        critic.zero_grad()

        # ======== Train Critic (Discriminator) ======== #

        encoder.freeze_parameters()
        decoder.freeze_parameters()
        critic.unfreeze_parameters()


        z_fake = torch.randn(images.size()[0], config.LATENT_DIMENSION) *  config.LATENT_SAMPLING_VARIANCE

        if torch.cuda.is_available():
            z_fake = z_fake.cuda()

        d_fake = critic(z_fake)

        z_real = encoder(images)
        d_real = critic(z_real)

        torch.log(d_fake).mean().unsqueeze(0).backward(mone)
        torch.log(1 - d_real).mean().unsqueeze(0).backward(mone)

        dis_optim.step()

        # ======== Train Generator ======== #

        encoder.unfreeze_parameters()
        decoder.unfreeze_parameters()
        critic.freeze_parameters()

        batch_size = images.size()[0]

        z_real = encoder(images)
        x_recon = decoder(z_real)
        d_real = critic(encoder(Variable(images.data)))

        recon_loss = criterion(x_recon, images)
        d_loss = config.LAMBDA * (torch.log(d_real)).mean()

        recon_loss.unsqueeze(0).backward(one)
        d_loss.unsqueeze(0).backward(mone)

        enc_optim.step()
        dec_optim.step()

        step += 1

        if (step + 1) % 300 == 0:
            print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
                  (epoch + 1, config.NUM_EPOCHS, step + 1, len(train_generator), recon_loss.data.item()))

    if (epoch + 1) % 1 == 0:
        batch_size = 104
        test_iter = iter(test_generator)
        test_data = next(test_iter)

        z_real = encoder(Variable(test_data[0]).cuda())
        reconst = decoder(torch.randn_like(z_real)).cpu().view(batch_size, 1, 28, 28)

        if not os.path.isdir('./data/reconst_images'):
            os.makedirs('data/reconst_images')

        save_image(test_data[0].view(batch_size, 1, 28, 28), './data/reconst_images/wae_gan_input.png')
        save_image(reconst.data, './data/reconst_images/wae_gan_images_%d.png' % (epoch + 1))
