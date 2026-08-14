"""MIT-licensed OCT diffusion denoiser adapted from DeweiHu/OCT_DDPM.

The upstream checkpoint and architecture are pinned to commit 8dfb2e6. The
network predicts diffusion noise at a fixed, calibrated timestep; it does not
generate a new retinal image from text or an unconstrained latent sample.

Copyright (c) 2023 Dewei Hu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


def _timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half_dim = embedding_dim // 2
    scale = math.log(10000) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -scale)
    values = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(values), torch.cos(values)], dim=1)
    return torch.nn.functional.pad(embedding, (0, 1, 0, 0)) if embedding_dim % 2 else embedding


def _swish(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


def _normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.interpolate(value, scale_factor=2.0, mode="nearest")
        return self.conv(value) if self.with_conv else value


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            return self.conv(torch.nn.functional.pad(value, (0, 1, 0, 1), mode="constant", value=0))
        return torch.nn.functional.avg_pool2d(value, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self, *, in_channels: int, out_channels: int | None = None,
        conv_shortcut: bool = False, dropout: float, temb_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = _normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, value: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(_swish(self.norm1(value)))
        hidden = hidden + self.temb_proj(_swish(temb))[:, :, None, None]
        hidden = self.conv2(self.dropout(_swish(self.norm2(hidden))))
        if self.in_channels != self.out_channels:
            value = self.conv_shortcut(value) if self.use_conv_shortcut else self.nin_shortcut(value)
        return value + hidden


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(value)
        query, key, projected = self.q(normalized), self.k(normalized), self.v(normalized)
        batch, channels, height, width = query.shape
        query = query.reshape(batch, channels, height * width).permute(0, 2, 1)
        key = key.reshape(batch, channels, height * width)
        weights = torch.softmax(torch.bmm(query, key) * (channels ** -0.5), dim=2).permute(0, 2, 1)
        attended = torch.bmm(projected.reshape(batch, channels, height * width), weights)
        return value + self.proj_out(attended.reshape(batch, channels, height, width))


class OCTDDPMNetwork(nn.Module):
    """Architecture matching the public OCT_DDPM checkpoint exactly."""

    def __init__(self):
        super().__init__()
        channels, out_channels, channel_multipliers = 32, 1, (1, 2, 2, 2)
        num_res_blocks, attention_resolutions, dropout = 2, (16,), 0.1
        self.ch = channels
        self.temb_ch = channels * 4
        self.num_resolutions = len(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.resolution = 512
        self.in_channels = 1
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(channels, self.temb_ch), nn.Linear(self.temb_ch, self.temb_ch),
        ])
        self.conv_in = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1)

        current_resolution = self.resolution
        input_multipliers = (1,) + channel_multipliers
        self.down = nn.ModuleList()
        block_in = channels
        for level in range(self.num_resolutions):
            blocks, attention = nn.ModuleList(), nn.ModuleList()
            block_in = channels * input_multipliers[level]
            block_out = channels * channel_multipliers[level]
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(
                    in_channels=block_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                ))
                block_in = block_out
                if current_resolution in attention_resolutions:
                    attention.append(AttnBlock(block_in))
            down = nn.Module()
            down.block, down.attn = blocks, attention
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                current_resolution //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=self.temb_ch, dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in,
            temb_channels=self.temb_ch, dropout=dropout,
        )

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            blocks, attention = nn.ModuleList(), nn.ModuleList()
            block_out = channels * channel_multipliers[level]
            skip_in = channels * channel_multipliers[level]
            for block_index in range(num_res_blocks + 1):
                if block_index == num_res_blocks:
                    skip_in = channels * input_multipliers[level]
                blocks.append(ResnetBlock(
                    in_channels=block_in + skip_in, out_channels=block_out,
                    temb_channels=self.temb_ch, dropout=dropout,
                ))
                block_in = block_out
                if current_resolution in attention_resolutions:
                    attention.append(AttnBlock(block_in))
            up = nn.Module()
            up.block, up.attn = blocks, attention
            if level != 0:
                up.upsample = Upsample(block_in, True)
                current_resolution *= 2
            self.up.insert(0, up)

        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, value: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if value.shape[2:] != (self.resolution, self.resolution):
            raise ValueError("OCT DDPM input must be 512 x 512")
        embedding = _timestep_embedding(timestep, self.ch)
        embedding = self.temb.dense[1](_swish(self.temb.dense[0](embedding)))
        history = [self.conv_in(value)]
        for level in range(self.num_resolutions):
            for block_index in range(self.num_res_blocks):
                hidden = self.down[level].block[block_index](history[-1], embedding)
                if self.down[level].attn:
                    hidden = self.down[level].attn[block_index](hidden)
                history.append(hidden)
            if level != self.num_resolutions - 1:
                history.append(self.down[level].downsample(history[-1]))
        hidden = self.mid.block_2(self.mid.attn_1(self.mid.block_1(history[-1], embedding)), embedding)
        for level in reversed(range(self.num_resolutions)):
            for block_index in range(self.num_res_blocks + 1):
                hidden = self.up[level].block[block_index](torch.cat([hidden, history.pop()], dim=1), embedding)
                if self.up[level].attn:
                    hidden = self.up[level].attn[block_index](hidden)
            if level != 0:
                hidden = self.up[level].upsample(hidden)
        return self.conv_out(_swish(self.norm_out(hidden)))


class OCTDiffusionDenoiser(nn.Module):
    def __init__(self, checkpoint: Path, timestep: int = 14):
        super().__init__()
        self.network = OCTDDPMNetwork()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.network.load_state_dict(state, strict=True)
        betas = torch.linspace(0.0001, 0.006, 100, dtype=torch.float32)
        self.register_buffer("alpha_cumprod", torch.cumprod(1 - betas, dim=0))
        self.timestep = timestep

    @staticmethod
    def _fit(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = image.shape
        ratio = min(512 / width, 512 / height)
        resized = cv2.resize(
            image, (round(width * ratio), round(height * ratio)),
            interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC,
        )
        top, left = (512 - resized.shape[0]) // 2, (512 - resized.shape[1]) // 2
        bottom, right = 512 - resized.shape[0] - top, 512 - resized.shape[1] - left
        fitted = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        return fitted, (top, bottom, left, right)

    def enhance(self, image: Image.Image) -> np.ndarray:
        gray = np.asarray(image.convert("L"))
        fitted, (top, bottom, left, right) = self._fit(gray)
        tensor = torch.from_numpy(1 + 2 * fitted.astype(np.float32) / 255)[None, None]
        tensor = tensor.to(self.alpha_cumprod.device)
        timestep = torch.tensor([self.timestep], device=tensor.device)
        predicted_noise = self.network(tensor, timestep)
        alpha = self.alpha_cumprod[self.timestep]
        estimate = tensor / torch.sqrt(alpha) - torch.sqrt(1 - alpha) / torch.sqrt(alpha) * predicted_noise
        output = np.clip((estimate[0, 0].float().cpu().numpy() - 1) * 127.5, 0, 255).astype(np.uint8)
        output = output[top:512 - bottom if bottom else 512, left:512 - right if right else 512]
        return cv2.resize(output, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
