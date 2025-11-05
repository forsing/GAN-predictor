# GAN predictor 
# PyTorch


"""        
=== System Information ===
Python version                 3.11.13        
macOS Apple                    Tahos 
Apple                          M1
"""


"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""


"""
svih 4506 izvlacenja
30.07.1985.- 04.11.2025.
"""


import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random


# reproducibilnost
SEED = 39

# Fiksiraj sve izvore nasumičnosti
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # ako koristiš više GPU-a

# Opciono: učini sve potpuno determinističkim (sporije, ali ponovljivo)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_and_normalize_data(csv_file="/data/loto7h_4506_k87.csv"):
    
    df = pd.read_csv(csv_file, encoding="utf-8")

    if "Num7" in df.columns and "num7" not in df.columns:
        df = df.rename(columns={"Num7": "num7"})
    for i in range(1, 7):
        if f"Num{i}" in df.columns and f"num{i}" not in df.columns:
            df = df.rename(columns={f"Num{i}": f"num{i}"})

    required_cols = [f"num{i}" for i in range(1, 7)] + ["num7"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"CSV中缺少必要列: {c}")

    all_draws = []
    for _, row in df.iterrows():
        reds = [row[f"num{i}"] for i in range(1, 7)]
        reds.sort()  # 升序
        blue = row["num7"]
        draw = reds + [blue]
        all_draws.append(draw)

    all_draws = np.array(all_draws, dtype=np.float32)  # shape (N,7)

    all_draws /= 39.0

    return all_draws


class Generator(nn.Module):
    def __init__(self, z_dim, data_dim=7):
        """
        data_dim=7
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, data_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        # (batch,7)
        out = self.net(z)  # shape: (batch_size,7)
        return out


class Discriminator(nn.Module):
    def __init__(self, data_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_gan_and_save(
    csv_file="/Users/milan/Desktop/GHQ/data/loto7h_4506_k87.csv",
    model_dir="gan_models",
    generator_file="G.pth",
    discriminator_file="D.pth",
    num_epochs=1000000,
    batch_size=32,
    z_dim=16
):
    """
    GAN
    """
    os.makedirs(model_dir, exist_ok=True)

    all_draws_norm = load_and_normalize_data(csv_file)
    data_tensor = torch.tensor(all_draws_norm, dtype=torch.float32)

    G = Generator(z_dim=z_dim, data_dim=7)
    D = Discriminator(data_dim=7)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        batch_size=32
        # data_tensor batch_size
        idx = np.random.randint(0, len(data_tensor), batch_size)
        real_data = data_tensor[idx]  # (batch_size,7)
        real_labels = torch.ones(batch_size, 1)

        z_dim=16
        z = torch.randn(batch_size, z_dim)
        fake_data = G(z)
        fake_labels = torch.zeros(batch_size, 1)

        d_optimizer.zero_grad()
        d_real = D(real_data)
        d_fake = D(fake_data.detach())
        d_loss_real = criterion(d_real, real_labels)
        d_loss_fake = criterion(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        d_fake_gen = D(fake_data)
        g_loss = criterion(d_fake_gen, real_labels)
        g_loss.backward()
        g_optimizer.step()

        if (epoch + 1) % 1 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] d_loss={d_loss.item():.4f}, g_loss={g_loss.item():.4f}"
            )

    torch.save(G.state_dict(), os.path.join(model_dir, generator_file))
    torch.save(D.state_dict(), os.path.join(model_dir, discriminator_file))
    print(f"GAN: G -> {generator_file}, D -> {discriminator_file}")


def load_gan_and_generate_norepeat(
    model_dir="gan_models",
    generator_file="G.pth",
    num_samples=1,
    z_dim=16,
    max_attempts=50
):
    """
    num_samples
    max_attempts
    shape=(num_samples,7)
    """
    G = Generator(z_dim=z_dim, data_dim=7)
    gen_path = os.path.join(model_dir, generator_file)
    G.load_state_dict(torch.load(gen_path))
    G.eval()

    results = []
    for _ in range(num_samples):
        for attempt in range(max_attempts):
            z = torch.randn(1, z_dim)
            with torch.no_grad():
                out = G(z).cpu().numpy()[0]  # shape (7,)

            reds = np.clip(np.round(out[:6] * 39).astype(int), 1, 39)
            blue = np.clip(int(round(out[6] * 39)), 1, 39)


            if len(set(reds)) == 6:
                row = np.concatenate([reds, [blue]])
                results.append(row)
                break
        else:
            # for attempt in range(max_attempts)
            row = fix_duplicates(np.concatenate([reds, [blue]]))
            results.append(row)

    return np.array(results, dtype=int)


def fix_duplicates(row7):
    """
    row7[:6]
    row7: shape(7,)
    """
    reds = list(row7[:6])
    unique_reds = []
    for r in reds:
        if r not in unique_reds:
            unique_reds.append(r)
        else:
            for cand in range(1, 40):
                if cand not in unique_reds:
                    unique_reds.append(cand)
                    break
    unique_reds = unique_reds[:6]
    unique_reds.sort()
    newrow = np.array(unique_reds + [row7[6]], dtype=int)
    return newrow


print()
if __name__ == "__main__":
    train_gan_and_save(
        csv_file="/loto7h_4506_k87.csv",
        model_dir="gan_models",
        generator_file="G.pth",
        discriminator_file="D.pth",
        num_epochs=1000000,
        batch_size=32,
        z_dim=16
    )

    samples = load_gan_and_generate_norepeat(
        model_dir="gan_models",
        generator_file="G.pth",
        num_samples=1,
        z_dim=16,
        max_attempts=50
    )
    for i, row in enumerate(samples):
        reds = row[:6]
        blue = row[6]
        print(f"Kombinacija {i+1}: {reds}, {blue}")
print()
"""
Kombinacija 1: [ 4  6 x x x 22], 27
"""

