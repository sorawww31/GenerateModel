#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import torchvision
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm


# In[2]:


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, t):
        N, C, _, _ = x.shape
        t = self.mlp(t)
        t = t.view(N, C, 1, 1)
        y = self.convs(x + t)
        return y


# In[3]:


class UNet(nn.Module):
    def __init__(self, out_ch=1, time_dim=100):
        super().__init__()
        self.time_dim = time_dim

        self.down1 = ConvBlock(1, 64, time_dim)
        self.down2 = ConvBlock(64, 128, time_dim)
        self.bot1 = ConvBlock(128, 256, time_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_dim)
        self.out = nn.Conv2d(64, out_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, t):
        v = pos_encoding(t, self.time_dim, x.device)  # 正弦波位置エンコーディング

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x


# In[4]:


def _pos_encoding(t, output_dim, device="cpu"):
    D = output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)  # ①i = [0, 1, ..., D-1]
    div_term = 10000 ** (i / D)

    # ②偶数の位置にsin、奇数の位置にcos
    v[0::2] = torch.sin(t / div_term[::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v


# 使用例
v = _pos_encoding(1, 16)
print(v.shape)  # (16,)


# In[5]:


def pos_encoding(ts, output_dim, device='cpu'):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)
    return v


# In[6]:


# 使用例
v = pos_encoding(torch.tensor([1, 2, 3]), 16)
print(v.shape)  # (3, 16)


# In[7]:


x = torch.randn(3, 64, 64)  # ダミーデータ
T = 1000
betas = torch.linspace(0.0001, 0.02, T)

for t in range(T):
    beta = betas[t]
    eps = torch.randn_like(x)  # xと同じ形状のガウスノイズを生成
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps


# In[8]:


# 画像の読み込み
#current_dir = os.path.dirname(os.path.abspath(__file__))
#file_path = os.path.join(current_dir, 'buiiki.png')
file_path = './DMmodel/buiiki.png'
image = plt.imread(file_path)
print(image.shape)  # (64, 64, 3)

# 画像の前処理を定義
preprocess = transforms.ToTensor()
x = preprocess(image)
print(x.shape)  # (3, 64, 64)


# In[9]:


def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)


# In[10]:


T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
imgs = []

for t in range(T):
    if t % 100 == 0:
        img = reverse_to_img(x)
        imgs.append(img)

    beta = betas[t]
    eps = torch.randn_like(x)
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps

# 10個の画像を2行5列で表示する
plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f'Noise: {i * 100}')
    plt.axis('off')
plt.show()


# In[11]:


x = torch.tensor([1, 2, 3, 4])
output = torch.cumprod(x, dim=0)

print(output,"i am here")


# In[12]:


img = plt.imread(file_path)
x = preprocess(img)

T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)

def add_noise(x_0, t, betas):
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar = alpha_bars[t]

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
    return x_t

# 時刻100のノイズ画像を生成
t = 100
x_t = add_noise(x, t, betas)

# 画像の表示
img = reverse_to_img(x_t)
plt.imshow(img)
plt.title(f'Noise: {t}')
plt.axis('off')
plt.show()


# In[13]:


class Diffuser:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        # ① ======================================
        t = t.to(device)
        alpha_bar = self.alpha_bars[t]  # (N,)
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # (N, 1, 1, 1)
        # ========================================
        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise
    def denoise(self, model, x, t):
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars[t-1]

        # ブロードキャストが正しく行われるための設定
        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        # ニューラルネットワークの実行
        model.eval()  # 評価モードにする
        with torch.no_grad():
            eps = model(x, t)
        model.train()

        mu = (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)

        if t[0] == 0:
            return mu
        else:
            noise = torch.randn_like(x, device=self.device)
            variance = (1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar)

            return mu + noise * torch.sqrt(variance)
    def reverse_to_img(self, x):
        x = x * 255.
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)
    
    def sample(self, model, x_shape=(20, 1, 28, 28)):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        # self.num_timestepsの値から1ずつ減らしていく逆順のループ
        for i in tqdm(range(self.num_timesteps)[::-1]):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images


# In[14]:


img_size = 28
batch_size = 128
timesteps = 1000
epochs = 10
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


def show_images(images, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.axis('off')  # 縦軸、横軸を非表示にする
            i += 1
    plt.show()

preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root="./data", download=True, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = Diffuser(timesteps)
model = UNet()
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

losses = []
cnt = 0
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = cnt+1

    #エポックごとにデータ生成して結果を確認したい場合は下記のコメントアウトをとる
    #images = diffuser.sample(model)
    #show_images(images)

    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        x = batch[0].to(device)
        t = torch.randint(0, timesteps, (len(x),), device=device).long()

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch {epoch} | Loss: {loss_avg}")

# 損失のグラフ
plt.plot(losses)

# 画像を生成
images = diffuser.sample(model)
show_images(images)

