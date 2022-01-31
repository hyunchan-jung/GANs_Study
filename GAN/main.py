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
parser.add_argument('--epochs', required=False, default=100, type=int)
parser.add_argument('--batch_size', required=False, default=128, type=int)
parser.add_argument('--hidden_dim', required=False, default=256, type=int)
parser.add_argument('--latent_dim', required=False, default=100, type=int)
parser.add_argument('--learning_rate', required=False, default=0.0002, type=float)
parser.add_argument('--num_workers', required=False, default=0, type=int)
args = parser.parse_args()


def load_data(data_path, batch_size, num_workers):
    """
    Load MNIST Dataset
    :return: Train, Test DataLoader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_dataset = torchvision.datasets.MNIST(data_path, download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(data_path, download=True, train=False, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader


def make_models(z_dim, h_dim):
    """
    :return: Generator, Discriminator
    """

    generator = nn.Sequential(
        nn.Linear(z_dim, h_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(h_dim, h_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(h_dim, 28 * 28),
        nn.Tanh()
    ).to(device)

    discriminator = nn.Sequential(
        nn.Linear(28 * 28, h_dim),
        nn.LeakyReLU(),
        nn.Dropout(0.1),
        nn.Linear(h_dim, h_dim),
        nn.LeakyReLU(),
        nn.Dropout(0.1),
        nn.Linear(h_dim, 1),
        nn.Sigmoid()
    ).to(device)

    return generator, discriminator


def run_epoch():
    """
    Adversarial G and D
    """

    generator.train()
    discriminator.train()

    for imgs, labels in train_dl:
        imgs, labels = imgs.view(-1, 28 * 28).to(device), labels.to(device)

        z = torch.randn(batch_size, z_dim).to(device)
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


def evaluate():
    p_real, p_fake = 0., 0.

    generator.eval()
    discriminator.eval()

    size = float(test_dl.batch_size * len(test_dl))
    for imgs, labels in test_dl:
        imgs, labels = imgs.view(-1, 28 * 28).to(device), labels.to(device)
        z = torch.randn(batch_size, z_dim).to(device)
        with torch.no_grad():
            p_real += discriminator(imgs).sum().item() / size
            p_fake += discriminator(generator(z)).sum().item() / size

    return p_real, p_fake


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = args.data_path
    batch_size = args.batch_size
    epochs = args.epochs
    h_dim = args.hidden_dim
    z_dim = args.latent_dim
    lr = args.learning_rate
    num_workers = args.num_workers

    train_dl, test_dl = load_data(data_path, batch_size, num_workers)

    generator, discriminator = make_models(z_dim, h_dim)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for i in range(epochs):
        run_epoch()
        p_real, p_fake = evaluate()
        print(f'epoch: {i}, p_real: {p_real:.6f}, p_fake: {p_fake:.6f}')
        z = torch.randn(16, z_dim).to(device)
        gen_imgs = make_grid(generator(z).reshape(-1, 1, 28, 28), nrow=8)
        save_image(gen_imgs, f'./Results/{i}.png')
