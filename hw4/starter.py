#!/usr/bin/env python3
"""
Starter code for pixel-space DDPM training on 64x64 RGB images.

Please implement the missing member functions in the PixelDDPM class. This
starter code already handles:
    a. dataset loading
    b. EMA helper
    c. sinusoidal timestep embedding
    d. U-Net denoiser
    e. training loop
    f. checkpointing
    g. saving sample generations in a 1x10 grid

The goal is to focus on the diffusion process itself, specifically:
    a. forward noising q(x_t | x_0)
    b. the training objective
    c. the reverse denoising process, including epsilon_theta
    d. the sampling process
"""

import math
import os
import copy
import random
import time
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image


# These are the hyper-parameters that I used to generate my samples.  Please
# feel free to experiment with different settings

DATA_DIR = Path('faces64')
OUT_DIR = Path('weights')
SAMPLES_DIR = OUT_DIR / 'samples'

IMAGE_SIZE = 64
CHANNELS = 3

T = 200
BETA_START = 1e-4
BETA_END = 2e-2
TIME_DIM = 256
BASE_CHANNELS = 96

EPOCHS = 20000
BATCH_SIZE = 64
LR = 5e-5
WEIGHT_DECAY = 1e-6
NUM_WORKERS = min(16, os.cpu_count() or 4)
SAVE_EVERY = 50

EMA_BETA = 0.995
EMA_START = 200

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ImageDataset loads training images from the faces64 directory.
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted([p for p in self.root.iterdir() if p.is_file()])
        if not self.paths:
            raise FileNotFoundError('No images found in {}'.format(self.root.resolve()))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img) if self.transform is not None else img


# EMA() implements exponential moving average to add stability during training
class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def update_average(self, old, new):
        return new if old is None else old * self.beta + (1.0 - self.beta) * new

    def update_model_average(self, ema_model, model):
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.data = self.update_average(ema_p.data, p.data)

    def step_ema(self, ema_model, model, step_start_ema=200):
        if self.step < step_start_ema:
            ema_model.load_state_dict(model.state_dict())
        else:
            self.update_model_average(ema_model, model)
        self.step += 1


# SinusoidalTimeEmbedding maps each timestep t to a richer feature vector so the
# U-Net can condition its predictions on where it is in the diffusion process.
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return F.pad(emb, (0, 1)) if self.dim % 2 else emb


# SelfAttention lets the U-Net relate distant spatial positions in the feature
# map, which helps it model long-range structure and global consistency.
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y = x.view(b, c, h * w).transpose(1, 2)
        y_ln = self.ln(y)
        a, _ = self.mha(y_ln, y_ln, y_ln)
        y = y + a
        y = y + self.ff(y)
        return y.transpose(1, 2).view(b, c, h, w)


# ResidualBlock applies two convolutional layers to extract and refine features.
# When residual=True, it also adds a skip connection so the U-Net can preserve
# information and train more stably.
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        self.residual = residual
        self.skip = None if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        g = 8 if out_ch >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
        )

    def forward(self, x):
        y = self.block(x)
        if not self.residual:
            return y
        return F.silu(y + (x if self.skip is None else self.skip(x)))


# Down and Up are the encoder and decoder blocks of the U-Net. They change
# spatial resolution, apply residual feature processing, and inject timestep information.
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_ch, in_ch, residual=True),
            ResidualBlock(in_ch, out_ch),
        )
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

    def forward(self, x, t):
        x = self.net(x)
        e = self.emb(t)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + e


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.net = nn.Sequential(
            ResidualBlock(in_ch + skip_ch, in_ch + skip_ch, residual=True),
            ResidualBlock(in_ch + skip_ch, out_ch),
        )
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

    def forward(self, x, skip, t):
        x = self.up(x)
        x = self.net(torch.cat([skip, x], dim=1))
        e = self.emb(t)[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + e


# UNet is the denoiser network used by the DDPM. It predicts the noise present
# in x_t while conditioning on the timestep embedding.
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, base=96):
        super().__init__()
        b = base
        self.time = SinusoidalTimeEmbedding(time_dim)

        self.inc = ResidualBlock(c_in, b)
        self.down1 = Down(b, b * 2, time_dim)
        self.down2 = Down(b * 2, b * 4, time_dim)
        self.attn1 = SelfAttention(b * 4)
        self.down3 = Down(b * 4, b * 4, time_dim)

        self.bot1 = ResidualBlock(b * 4, b * 8)
        self.bot2 = ResidualBlock(b * 8, b * 8, residual=True)
        self.bot3 = ResidualBlock(b * 8, b * 4)

        self.up1 = Up(b * 4, b * 4, b * 2, time_dim)
        self.attn2 = SelfAttention(b * 2)
        self.up2 = Up(b * 2, b * 2, b, time_dim)
        self.up3 = Up(b, b, b, time_dim)

        self.out = nn.Conv2d(b, c_out, 1)

    def forward(self, x, t):
        t = self.time(t)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x3 = self.attn1(self.down2(x2, t))
        x4 = self.down3(x3, t)

        x4 = self.bot3(self.bot2(self.bot1(x4)))

        x = self.attn2(self.up1(x4, x3, t))
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        return self.out(x)


class PixelDDPM(nn.Module):
    def __init__(self, T=200, beta_start=1e-4, beta_end=2e-2, time_dim=256, base_channels=96):
        super().__init__()
        self.T = T
        self.eps_model = UNet(CHANNELS, CHANNELS, time_dim, base_channels)

        # TODO:
        # 1. Create the beta schedule with torch.linspace(beta_start, beta_end, T)
        # 2. Compute alphas = 1 - betas
        # 3. Compute alpha_bars = cumulative product of alphas
        # 4. Compute helper tensors used in q_sample and p_sample
        # 5. Store them with self.register_buffer(...)
        # 6. Compute posterior_variance for reverse sampling

        raise NotImplementedError('Students should implement PixelDDPM.__init__')

    def q_sample(self, x0, t, noise=None):
        # TODO:
        # Implement forward diffusion:
        #   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # Return x_t and the noise used.
        raise NotImplementedError('Students should implement PixelDDPM.q_sample')

    def forward(self, x0):
        # TODO:
        # 1. Sample a random timestep for each image in the batch
        # 2. Call q_sample(x0, t)
        # 3. Predict the noise with self.eps_model(x_t, t)
        # 4. Return MSE(predicted_noise, true_noise)
        raise NotImplementedError('Students should implement PixelDDPM.forward')

    @torch.no_grad()
    def p_sample(self, xt, t_scalar, model):
        # TODO:
        # Implement one reverse diffusion step.
        # Use the DDPM reverse mean formula and add Gaussian noise
        # unless t_scalar == 0.
        raise NotImplementedError('Students should implement PixelDDPM.p_sample')

    @torch.no_grad()
    def sample(self, n, model):
        # TODO:
        # Start from pure Gaussian noise and repeatedly call p_sample
        # from timestep T-1 down to 0.
        raise NotImplementedError('Students should implement PixelDDPM.sample')

    @torch.no_grad()
    def save_sample_grid(self, epoch, out_dir, model, n=10):
        # TODO:
        # Generate n images with self.sample(...)
        # Convert them from [-1,1] to [0,1]
        # Save them as a 1x10 strip using save_image(..., nrow=10)
        raise NotImplementedError('Students should implement PixelDDPM.save_sample_grid')


def main():
    random.seed(161)
    torch.manual_seed(161)
    torch.cuda.manual_seed_all(161)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print('Using device:', DEVICE)
    if DEVICE.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print('GPU:', torch.cuda.get_device_name(0))
        print('VRAM: {:.2f} GB'.format(props.total_memory / (1024**3)))

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = ImageDataset(DATA_DIR, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=NUM_WORKERS if len(dataset) > 16 else 0,
        pin_memory=(DEVICE.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0 and len(dataset) > 16),
    )

    ddpm = PixelDDPM(T, BETA_START, BETA_END, TIME_DIM, BASE_CHANNELS).to(DEVICE)
    ema_model = copy.deepcopy(ddpm.eps_model).eval().to(DEVICE)
    for p in ema_model.parameters():
        p.requires_grad = False

    ema = EMA(EMA_BETA)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print('Total training images:', len(dataset))
    print('T = {}, LR = {}, BASE_CHANNELS = {}'.format(T, LR, BASE_CHANNELS))
    print('Sample grid every {} epochs'.format(SAVE_EVERY))

    ddpm.eval()
    ddpm.save_sample_grid(0, SAMPLES_DIR, ema_model)
    print('Saved sample grid for epoch 0')

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        ddpm.train()
        running_loss, n_seen = 0.0, 0

        for images in loader:
            images = images.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = ddpm(images)
            loss.backward()
            optimizer.step()

            ema.step_ema(ema_model, ddpm.eps_model, step_start_ema=EMA_START)

            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        epoch_loss = running_loss / max(n_seen, 1)
        epoch_time = time.time() - start

        print(
            'Epoch [{:06d}/{}] loss={:.8f} time_per_epoch={:.2f}s'.format(
                epoch, EPOCHS, epoch_loss, epoch_time
            )
        )

        ckpt = {
            'epoch': epoch,
            'ddpm_state_dict': ddpm.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(ckpt, OUT_DIR / 'pixel_diffusion_latest.pt')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, OUT_DIR / 'pixel_diffusion_best.pt')

        if epoch % SAVE_EVERY == 0:
            ddpm.eval()
            ddpm.save_sample_grid(epoch, SAMPLES_DIR, ema_model)
            print('Saved sample grid for epoch {}'.format(epoch))

    print('\nTraining complete.')
    print('Best diffusion loss: {:.8f}'.format(best_loss))
    print('Saved checkpoints to:', OUT_DIR.resolve())


if __name__ == '__main__':
    main()
