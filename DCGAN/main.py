import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True)
parser.add_argument('--hidden_dim', required=False, default=64, type=int)
parser.add_argument('--epochs', required=False, default=100, type=int)
parser.add_argument('--batch_size', required=False, default=128, type=int)
parser.add_argument('--num_workers', required=False, default=0, type=int)
parser.add_argument('--load_weights', required=False, default=None)
args = parser.parse_args()


def load_data(data_path, batch_size, num_workers):
    """
    Load MNIST Dataset
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_dataset = torchvision.datasets.MNIST(data_path, download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(data_path, download=True, train=False, transform=transform)
    dataset = train_dataset + test_dataset

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


def make_models():
    """
    :return: Generator, Discriminator
    """

    generator = nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        nn.ConvTranspose2d(ngf, 1, 4, 2, 3, bias=False),
        nn.Tanh()
    ).to(device)

    discriminator = nn.Sequential(
        nn.Conv2d(1, ndf, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 3, 2, 1),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 0),
        nn.Sigmoid()
    ).to(device)

    return generator, discriminator


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def run_epoch():
    """
    Adversarial G and D
    """

    generator.train()
    discriminator.train()

    history = {'D_loss': [],
               'G_loss': []}

    for imgs, _ in dataloader:
        imgs = imgs.to(device)

        z = torch.randn(batch_size, nz, 1, 1).to(device)
        p_real = discriminator(imgs)
        gen_out = generator(z)

        # Generator
        optimizer_g.zero_grad()
        p_fake = discriminator(gen_out)
        loss_g = criterion(p_fake, torch.ones_like(p_fake))
        loss_g.backward()
        optimizer_g.step()

        # Discriminator
        optimizer_d.zero_grad()
        p_fake = discriminator(gen_out.detach())
        loss_d_real = criterion(p_real, torch.ones_like(p_real))
        loss_d_fake = criterion(p_fake, torch.zeros_like(p_fake))
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        history['D_loss'].append(loss_d.item())
        history['G_loss'].append(loss_g.item())

    return history


def save_weights(fp):
    state_dicts = {'G_state_dict': generator.state_dict(),
                   'D_state_dict': discriminator.state_dict()}
    torch.save(state_dicts, fp)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = args.data_path
    batch_size = args.batch_size
    epochs = args.epochs
    num_workers = args.num_workers
    load_weights = args.load_weights
    hidden_dim = args.hidden_dim
    ngf, ndf = hidden_dim, hidden_dim
    nz = 100
    lr = 0.0002
    beta1 = 0.5

    dataloader = load_data(data_path, batch_size, num_workers)

    generator, discriminator = make_models()
    if load_weights is not None:
        checkpoint = torch.load(load_weights)
        generator.load_state_dict(checkpoint['G_state_dict'])
        discriminator.load_state_dict(checkpoint['D_state_dict'])
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    for i in range(epochs):
        history = run_epoch()
        loss_d = np.mean(history['D_loss'])
        loss_g = np.mean(history['G_loss'])
        print(f'epoch: {i}, loss_d: {loss_d:.6f}, loss_g: {loss_g:.6f}')

        if i % 10 == 0:
            save_weights(f'./Weights/{i}.tar')

        z = torch.randn(64, nz, 1, 1).to(device)
        gen_imgs = make_grid(generator(z).reshape(-1, 1, 28, 28), nrow=8)
        save_image(gen_imgs, f'./Results/{i}.png')
