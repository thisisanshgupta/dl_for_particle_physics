import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from tqdm.auto import tqdm

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not detected.")

MAX_JETS = 10_000 #10k jet samples
MAX_PARTICLES = 128

print("Loading JetClass-II...")
print(f"Target number of jets: {MAX_JETS}")

dataset = load_dataset(
    "jet-universe/jetclass2",
    split="train",
    streaming=True
)

dataset = dataset.take(MAX_JETS)

print(f"Using only {MAX_JETS} jets")
print("Dataset loaded in streaming mode.")


def wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


jets = []

for i, event in enumerate(dataset):
    if i >= MAX_JETS:
        break

    deta = np.asarray(event["part_deta"], dtype=np.float32)
    dphi = np.asarray(event["part_dphi"], dtype=np.float32)
    px = np.asarray(event["part_px"], dtype=np.float32)
    py = np.asarray(event["part_py"], dtype=np.float32)

    pt = np.sqrt(px**2 + py**2)

    order = np.argsort(pt)[::-1]

    pt = pt[order]
    deta = deta[order]
    dphi = dphi[order]

    n = min(len(pt), MAX_PARTICLES)

    jet = np.zeros((MAX_PARTICLES, 3), dtype=np.float32)

    jet[:n, 0] = pt[:n]
    jet[:n, 1] = deta[:n]
    jet[:n, 2] = dphi[:n]

    jets.append(jet.reshape(-1))

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{MAX_JETS} jets")

jets = np.asarray(jets, dtype=np.float32)

print()
print("Finished!")
print("Jets:", jets.shape)


PT_MAX = 1000.0
DETA_MAX = 2.0
DPHI_MAX = np.pi

x = jets.reshape(-1, MAX_PARTICLES, 3).copy()

x[:, :, 0] = np.clip(x[:, :, 0] / PT_MAX, 0.0, 1.0)
x[:, :, 1] = np.clip((x[:, :, 1] + DETA_MAX) / (2 * DETA_MAX), 0.0, 1.0)
x[:, :, 2] = np.clip((x[:, :, 2] + DPHI_MAX) / (2 * DPHI_MAX), 0.0, 1.0)

x = x.reshape(len(x), 384)

print("Final training tensor:", x.shape)
print("Minimum:", x.min())
print("Maximum:", x.max())

tensor_x = torch.tensor(x, dtype=torch.float32)
dataset_torch = TensorDataset(tensor_x)

loader = DataLoader(
    dataset_torch,
    batch_size=256,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

print(f"Training on {len(dataset_torch)} jets")
print(x.shape)
print(x.dtype)
print(x.min(), x.max())


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_DIM = 384
HIDDEN_DIM = 400
LATENT_DIM = 200

LR = 1e-3
BETA = 1e-4


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_variance = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.fc_input(x))
        h = self.LeakyReLU(self.fc_input2(h))
        mean = self.fc_mean(h)
        log_variance = self.fc_variance(h)
        return mean, log_variance


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.fc_hidden(x))
        h = self.LeakyReLU(self.fc_hidden2(h))
        x_hat = torch.sigmoid(self.fc_output(h))
        return x_hat


class VariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.Encoder = encoder
        self.Decoder = decoder

    def reparameterization(self, mean, variance):
        epsilon = torch.randn_like(variance)
        z = mean + variance * epsilon
        return z

    def forward(self, x):
        mean, log_variance = self.Encoder(x)
        variance = torch.exp(0.5 * log_variance)
        z = self.reparameterization(mean, variance)
        x_hat = self.Decoder(z)
        return x_hat, mean, log_variance


def loss_function(x_hat, x, mean, log_variance):
    reconstruction_loss = nn.functional.binary_cross_entropy(
        x_hat, x, reduction="mean"
    )
    kl_divergence = -0.5 * torch.mean(
        1 + log_variance - mean.pow(2) - log_variance.exp()
    )
    total_loss = reconstruction_loss + BETA * kl_divergence
    return total_loss


encoder = Encoder(X_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, X_DIM)
VAE = VariationalAutoencoder(encoder, decoder).to(DEVICE)

optimizer = Adam(VAE.parameters(), lr=LR)

print("========================================")
print("Particle Physics VAE")
print("========================================")
print(f"Device:          {DEVICE}")
print(f"Input dimension: {X_DIM}")
print(f"Hidden dimension: {HIDDEN_DIM}")
print(f"Latent dimension: {LATENT_DIM}")
print(f"Learning rate:   {LR}")
print(f"KL beta:         {BETA}")
print(f"Parameters:      {sum(p.numel() for p in VAE.parameters()):,}")
print("========================================")


NUM_EPOCHS = 50

CHECKPOINT_DIR = "/content/vae_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

training_history = []

print("Starting training...")
print(f"Device: {DEVICE}")
print(f"Number of jets: {len(dataset_torch)}")
print(f"Batch size: {loader.batch_size}")
print(f"Epochs: {NUM_EPOCHS}\n")

for epoch in range(NUM_EPOCHS):
    VAE.train()

    epoch_loss = 0.0
    epoch_reconstruction = 0.0
    epoch_kl = 0.0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for (batch,) in progress_bar:
        batch = batch.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        x_hat, mean, log_variance = VAE(batch)

        reconstruction_loss = nn.functional.binary_cross_entropy(
            x_hat, batch, reduction="mean"
        )

        kl_divergence = -0.5 * torch.mean(
            1 + log_variance - mean.pow(2) - log_variance.exp()
        )

        loss = reconstruction_loss + BETA * kl_divergence

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_reconstruction += reconstruction_loss.item()
        epoch_kl += kl_divergence.item()

        progress_bar.set_postfix(loss=f"{loss.item():.5f}")

    epoch_loss /= len(loader)
    epoch_reconstruction /= len(loader)
    epoch_kl /= len(loader)

    training_history.append({
        "loss": epoch_loss,
        "reconstruction": epoch_reconstruction,
        "kl": epoch_kl
    })

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Total: {epoch_loss:.6f} | "
        f"Reconstruction: {epoch_reconstruction:.6f} | "
        f"KL: {epoch_kl:.6f}"
    )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"vae_epoch_{epoch + 1}.pt"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": VAE.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "reconstruction_loss": epoch_reconstruction,
            "kl_divergence": epoch_kl,
        },
        checkpoint_path
    )

print("\nTraining complete.")


epochs = range(1, len(training_history) + 1)
losses = [h["loss"] for h in training_history]
reconstruction_losses = [h["reconstruction"] for h in training_history]
kl_losses = [h["kl"] for h in training_history]

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, label="Total loss")
plt.plot(epochs, reconstruction_losses, label="Reconstruction")
plt.plot(epochs, kl_losses, label="KL divergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training")
plt.legend()
plt.grid(alpha=0.2)
plt.show()


VAE.eval()

with torch.no_grad():
    sample_batch = next(iter(loader))[0].to(DEVICE)
    reconstructed, mean, log_variance = VAE(sample_batch)

print("Input shape:")
print(sample_batch.shape)
print()
print("Reconstruction shape:")
print(reconstructed.shape)
print()
print("Latent mean shape:")
print(mean.shape)


def denormalize_jets(x):
    x = x.reshape(-1, MAX_PARTICLES, 3).copy()

    x[:, :, 0] *= PT_MAX
    x[:, :, 1] = x[:, :, 1] * (2 * DETA_MAX) - DETA_MAX
    x[:, :, 2] = x[:, :, 2] * (2 * DPHI_MAX) - DPHI_MAX

    return x


real_jets = denormalize_jets(sample_batch.cpu().numpy())
reconstructed_jets = denormalize_jets(reconstructed.cpu().numpy())

print("Real jets:", real_jets.shape)
print("Reconstructed jets:", reconstructed_jets.shape)


VAE.eval()
N_GENERATE = 1000

with torch.no_grad():
    z = torch.randn(N_GENERATE, LATENT_DIM, device=DEVICE)
    generated_normalized = VAE.Decoder(z)

generated_jets = denormalize_jets(generated_normalized.cpu().numpy())

print("Generated jets:", generated_jets.shape)

np.save("/content/generated_jets.npy", generated_jets)
print("Saved to /content/generated_jets.npy")

jet_id = 0
jet = generated_jets[jet_id]

pt = jet[:, 0]
deta = jet[:, 1]
dphi = jet[:, 2]

mask = pt > 0

plt.figure(figsize=(7, 7))
plt.scatter(
    deta[mask],
    dphi[mask],
    s=np.maximum(pt[mask] / 5, 2),
    alpha=0.7
)
plt.xlabel(r"$\Delta\eta$")
plt.ylabel(r"$\Delta\phi$")
plt.title("Generated Jet")
plt.grid(alpha=0.2)
plt.show()

real_pt = real_jets[:, :, 0].flatten()
generated_pt = generated_jets[:, :, 0].flatten()

real_pt = real_pt[real_pt > 0]
generated_pt = generated_pt[generated_pt > 0]

plt.figure(figsize=(8, 5))
plt.hist(real_pt, bins=60, density=True, alpha=0.6, label="Real")
plt.hist(generated_pt, bins=60, density=True, alpha=0.6, label="Generated")
plt.xlabel(r"Constituent $p_T$")
plt.ylabel("Density")
plt.title("Real vs Generated Constituent $p_T$")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

real_deta = real_jets[:, :, 1].flatten()
generated_deta = generated_jets[:, :, 1].flatten()

real_deta = real_deta[real_pt.size * 0:]
real_flat = real_jets.reshape(-1, 3)
generated_flat = generated_jets.reshape(-1, 3)

real_mask = real_flat[:, 0] > 0
generated_mask = generated_flat[:, 0] > 0

real_deta = real_flat[real_mask, 1]
generated_deta = generated_flat[generated_mask, 1]

real_dphi = real_flat[real_mask, 2]
generated_dphi = generated_flat[generated_mask, 2]

plt.figure(figsize=(8, 5))
plt.hist(real_deta, bins=60, density=True, alpha=0.6, label="Real")
plt.hist(generated_deta, bins=60, density=True, alpha=0.6, label="Generated")
plt.xlabel(r"$\Delta\eta$")
plt.ylabel("Density")
plt.title(r"Real vs Generated $\Delta\eta$")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(real_dphi, bins=60, density=True, alpha=0.6, label="Real")
plt.hist(generated_dphi, bins=60, density=True, alpha=0.6, label="Generated")
plt.xlabel(r"$\Delta\phi$")
plt.ylabel("Density")
plt.title(r"Real vs Generated $\Delta\phi$")
plt.legend()
plt.grid(alpha=0.2)
plt.show()


VAE.eval()

with torch.no_grad():
    latent_mean, latent_logvar = VAE.Encoder(tensor_x.to(DEVICE))

latent = latent_mean.cpu().numpy()

print("Latent representation:", latent.shape)

pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent)

print("Explained variance:", pca.explained_variance_ratio_)

plt.figure(figsize=(8, 7))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], s=4, alpha=0.4)
plt.xlabel("Latent PC 1")
plt.ylabel("Latent PC 2")
plt.title("VAE Latent Space")
plt.grid(alpha=0.2)
plt.show()

print("Real Δη:")
print("min =", real_deta.min())
print("max =", real_deta.max())

print("\nGenerated Δη:")
print("min =", generated_deta.min())
print("max =", generated_deta.max())

print("\nReal Δφ:")
print("min =", real_dphi.min())
print("max =", real_dphi.max())

print("\nGenerated Δφ:")
print("min =", generated_dphi.min())
print("max =", generated_dphi.max())

print("Real constituents:", len(real_deta))
print("Generated constituents:", len(generated_deta))

plt.figure(figsize=(8, 5))
plt.hist(real_pt, bins=80, density=True, alpha=0.6, label="Real")
plt.hist(generated_pt, bins=80, density=True, alpha=0.6, label="Generated")
plt.xlabel(r"Constituent $p_T$")
plt.ylabel("Density")
plt.title(r"Real vs Generated Constituent $p_T$")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

real_R = np.sqrt(real_deta**2 + real_dphi**2)
generated_R = np.sqrt(generated_deta**2 + generated_dphi**2)

plt.figure(figsize=(8, 5))
plt.hist(real_R, bins=60, density=True, alpha=0.6, label="Real")
plt.hist(generated_R, bins=60, density=True, alpha=0.6, label="Generated")
plt.xlabel(r"$\Delta R$")
plt.ylabel("Density")
plt.title(r"Real vs Generated $\Delta R$")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
