#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# In[ ]:


# ハイパラ
input_dim = 784  # 画像データxのサイズ（MNIST画像は28x28=784）
hidden_dim = 200  # ニューラルネットワークの中間層の次元数
latent_dim = 20  # 潜在変数ベクトルzの次元数
epochs = 30
learning_rate = 3e-4
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_sigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.linear(x))
        mu = self.linear_mu(x)
        sigma = self.linear_sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat


# In[4]:


def reparameter(mu, sigma):
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


# In[5]:


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = reparameter(mu, sigma)
        x_hat = self.decoder(z)

        batch_size = len(x)
        L1 = F.mse_loss(x, x_hat, reduction="sum")
        L2 = torch.sum(mu**2 + sigma**2 - torch.log(sigma**2) - 1)
        return (L1 + L2) / batch_size


# In[ ]:


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(torch.flatten)]  # 画像をフラットに
)
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ②モデルとオプティマイザ
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

# ③学習ループ
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for x, label in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(loss_avg)


# In[ ]:


with torch.no_grad():  # ①
    sample_size = 64
    z = torch.randn(sample_size, latent_dim)  # ②
    x = model.decoder(z)  # ③
    generated_images = x.view(sample_size, 1, 28, 28)  # ④

# ⑤
grid_img = torchvision.utils.make_grid(
    generated_images, nrow=8, padding=2, normalize=True
)

plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.show()


# In[ ]:
