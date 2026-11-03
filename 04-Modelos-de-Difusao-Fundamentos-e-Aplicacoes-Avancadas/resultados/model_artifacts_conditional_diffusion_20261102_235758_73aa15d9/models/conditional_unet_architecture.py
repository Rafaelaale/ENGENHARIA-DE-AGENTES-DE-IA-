
# Arquitetura exportada do Projeto 04.

import math
import os
import random

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset

EXT = {".png", ".jpg", ".jpeg", ".bmp"}

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    emb = t[:, None].float() * freqs[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=1)

class LabelEmbedding(nn.Module):

    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels):
        return self.embedding(labels)

class ConditionalTimeEmbedding(nn.Module):

    def __init__(self, time_dim, label_dim):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.label_mlp = nn.Sequential(nn.Linear(label_dim, label_dim * 2), nn.SiLU(), nn.Linear(label_dim * 2, time_dim))
        self.fusion = nn.Sequential(nn.Linear(time_dim * 2, time_dim), nn.SiLU())

    def forward(self, t, label_emb):
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        l_emb = self.label_mlp(label_emb)
        combined = torch.cat((t_emb, l_emb), dim=1)
        return self.fusion(combined)

class ConditionalResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim, groups=8, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.act1 = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, cond_emb):
        residual = self.skip(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        cond = self.time_proj(cond_emb)[:, :, None, None]
        h = h + cond
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        h = self.act2(h)
        return (h + residual) / math.sqrt(2)

class ConditionalDown(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = ConditionalResidualBlock(in_ch, out_ch, time_dim)
        self.block2 = ConditionalResidualBlock(out_ch, out_ch, time_dim)
        self.pool = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, cond_emb):
        x = self.block1(x, cond_emb)
        x = self.block2(x, cond_emb)
        return (x, self.pool(x))

class ConditionalUp(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.block1 = ConditionalResidualBlock(out_ch + skip_ch, out_ch, time_dim)
        self.block2 = ConditionalResidualBlock(out_ch, out_ch, time_dim)

    def forward(self, x, skip, cond_emb):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = self.block1(x, cond_emb)
        x = self.block2(x, cond_emb)
        return x

class ConditionalUNet(nn.Module):

    def __init__(self, in_channels=1, base_channels=64, num_classes=3, time_dim=256, label_dim=128, dropout_prob=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.null_label = num_classes
        self.dropout_prob = dropout_prob
        self.label_embedding = LabelEmbedding(num_classes + 1, label_dim)
        self.time_embedding = ConditionalTimeEmbedding(time_dim, label_dim)
        self.down1 = ConditionalDown(in_channels, base_channels, time_dim)
        self.down2 = ConditionalDown(base_channels, base_channels * 2, time_dim)
        self.down3 = ConditionalDown(base_channels * 2, base_channels * 4, time_dim)
        self.down4 = ConditionalDown(base_channels * 4, base_channels * 8, time_dim)
        self.bot1 = ConditionalResidualBlock(base_channels * 8, base_channels * 16, time_dim)
        self.bot2 = ConditionalResidualBlock(base_channels * 16, base_channels * 8, time_dim)
        self.up4 = ConditionalUp(base_channels * 8, base_channels * 8, base_channels * 4, time_dim)
        self.up3 = ConditionalUp(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up2 = ConditionalUp(base_channels * 2, base_channels * 2, base_channels, time_dim)
        self.up1 = ConditionalUp(base_channels, base_channels, base_channels, time_dim)
        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t, labels=None):
        if labels is None:
            ids = torch.full((x.size(0),), self.null_label, dtype=torch.long, device=x.device)
        else:
            ids = labels.to(device=x.device, dtype=torch.long).clone()
            if self.training:
                mask = torch.rand(ids.size(0), device=x.device) < self.dropout_prob
                ids[mask] = self.null_label
        label_emb = self.label_embedding(ids)
        cond_emb = self.time_embedding(t, label_emb)
        s1, p1 = self.down1(x, cond_emb)
        s2, p2 = self.down2(p1, cond_emb)
        s3, p3 = self.down3(p2, cond_emb)
        s4, p4 = self.down4(p3, cond_emb)
        b = self.bot1(p4, cond_emb)
        b = self.bot2(b, cond_emb)
        u4 = self.up4(b, s4, cond_emb)
        u3 = self.up3(u4, s3, cond_emb)
        u2 = self.up2(u3, s2, cond_emb)
        u1 = self.up1(u2, s1, cond_emb)
        return self.out(u1)
