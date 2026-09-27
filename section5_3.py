#!/usr/bin/env python3
r"""
Experiment 5.3: Exploratory Analysis of Latent vs. Ambient Density Smoothing on MNIST

Overview:
---------
This script runs exploratory numerical experiments to investigate the practical behavior 
of the latent density smoothing on high-dimensional image data (MNIST).

Standard ambient kernel smoothing in high dimensions suffers from off-manifold 
noise inflation, where variance reduction is heavily penalized by an ambient dimension 
scaling of O(sigma * sqrt(D - m)). Latent density smoothing aims to bypass this penalty 
by performing heat diffusion in a lower-dimensional latent bottleneck via a geometry-
preserving autoencoder (Lee et al., 2025) before decoding back to ambient space.

Experimental Setup:
-------------------
1. Data Sample: A small exploratory sub-sample of N = 20 images drawn from MNIST.
2. Architecture: Geometry-preserving encoder/decoder pair trained to balance 
   reconstruction fidelity, metric distortion, and off-surface expansion.
3. Comparison:
   - Ambient Kernel Smoothing: Direct Gaussian heat smoothing in ambient space (D = 784).
   - Latent Density Smoothing: Heat kernel smoothing injected in the latent bottleneck (R^m).
4. Tracking: Tracks changes in the squared Wasserstein-2 (W_2^2) distance relative 
   to the target distribution across a sweep of smoothing bandwidths sigma.

Exploratory Objectives:
-----------------------
- Observe empirical trends of ambient noise inflation in high-dimensional image space.
- Explore the preliminary effectiveness of latent density smoothing in small-sample regimes.
- Map the empirical trajectory of squared Wasserstein-2 distance across bandwidths sigma 
  to gather insights for theoretical refinements and hyperparameter tuning.

The code is written together with ChatGPT
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 0. Setup & Hyperparameters
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LATENT_DIM = 15     # Latent bottleneck dimension d
AMBIENT_DIM = 784   # Ambient dimension D (28x28)
N_SUBSAMPLE = 20    # Data-scarce sample size n
N_EVAL = 5000       # Number of evaluation points for OT
SIGMA_TRAIN = 0.05  # Latent noise level for decoder training

# Increased epochs since 1 epoch on n=20 is only 1 gradient step
EPOCHS_ENC = 50000   # Training epochs for encoder on n=20
EPOCHS_DEC = 50000   # Training epochs for decoder on n=20

RESULTS_DIR = "./results_only_20"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Subsample-specific checkpoint paths
ENC_CKPT = os.path.join(RESULTS_DIR, f"gpe_encoder_subsample_n{N_SUBSAMPLE}_d{LATENT_DIM}.pt")
DEC_CKPT = os.path.join(RESULTS_DIR, f"gpe_decoder_subsample_n{N_SUBSAMPLE}_d{LATENT_DIM}.pt")


# ==========================================
# 1. Higher-Capacity GPE Architectures (Softplus)
# ==========================================
class GPEEncoder(nn.Module):
    def __init__(self, in_dim=784, out_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.Softplus(),
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class GPEDecoder(nn.Module):
    def __init__(self, in_dim=24, out_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Softplus(),
            nn.Linear(128, 256),
            nn.Softplus(),
            nn.Linear(256, 512),
            nn.Softplus(),
            nn.Linear(512, out_dim),
            nn.Sigmoid()  # Pixel intensity in [0, 1]
        )

    def forward(self, z):
        return self.net(z)


# ==========================================
# 2. Theoretical Metric Evaluation Function
# ==========================================
def evaluate_theoretical_properties(
    encoder, 
    decoder, 
    X_sub, 
    X_full=None, 
    intrinsic_dim=10, 
    eval_samples=100, 
    k_nn=15
):
    """
    Computes diagnostic theoretical properties on X_sub:
      1. delta: RMS reconstruction error
      2. eps_iso: Tangential isometry defect on local tangent space T_x M
      3. L_orth: Orthogonal expansion factor in normal latent space
    """
    encoder.eval()
    decoder.eval()
    device = X_sub.device
    n, D = X_sub.shape
    d = encoder.net[-1].out_features

    if X_full is None:
        X_full = X_sub

    if not isinstance(X_full, torch.Tensor):
        X_full_tensor = torch.tensor(X_full, device=device, dtype=torch.float32)
    else:
        X_full_tensor = X_full.to(device)

    # 1. Reconstruction Errors (\delta)
    with torch.no_grad():
        z_sub = encoder(X_sub)
        x_rec = decoder(z_sub)

        l2_errors = torch.sqrt(torch.sum((X_sub - x_rec)**2, dim=1))
        delta_rms = torch.sqrt(torch.mean(l2_errors**2)).item()
        delta_pixel = delta_rms / np.sqrt(D)

        mean_data_norm = torch.sqrt(torch.mean(torch.sum(X_sub**2, dim=1))).item()
        delta_rel = delta_rms / (mean_data_norm + 1e-8)
        delta = delta_rel

    print(f"Reconstruction Error (delta_rms):    {delta_rms:.5f}")
    print(f"Per-Pixel RMS Error (delta_pixel):   {delta_pixel:.5f}")
    print(f"Relative Error (delta_rel):          {delta_rel * 100:.2f}%")

    X_full_np = X_full_tensor.cpu().numpy()
    X_sub_np = X_sub.cpu().numpy()

    k = min(k_nn, len(X_full_np))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_full_np)
    num_eval = min(eval_samples, n)

    eps_iso_list = []
    L_orth_list = []

    for i in range(num_eval):
        x_i = X_sub[i:i+1]

        J_E = torch.autograd.functional.jacobian(encoder, x_i).squeeze()

        with torch.no_grad():
            z_i = encoder(x_i)
        J_D = torch.autograd.functional.jacobian(decoder, z_i).squeeze()

        J_DE = torch.matmul(J_D, J_E)

        indices = nbrs.kneighbors(X_sub_np[i:i+1], return_distance=False)[0]
        local_pts = X_full_tensor[indices]
        local_centered = local_pts - torch.mean(local_pts, dim=0, keepdim=True)

        _, _, V_local = torch.linalg.svd(local_centered, full_matrices=False)
        V_tan = V_local[:intrinsic_dim, :].T

        J_DE_tangent = torch.matmul(V_tan.T, torch.matmul(J_DE, V_tan))
        I_m = torch.eye(intrinsic_dim, device=device)
        eps_iso_i = torch.linalg.matrix_norm(J_DE_tangent - I_m, ord=2).item()
        eps_iso_list.append(eps_iso_i)

        U_E, _, _ = torch.linalg.svd(J_E)
        if intrinsic_dim < d:
            U_orth = U_E[:, intrinsic_dim:]
            J_D_orth = torch.matmul(J_D, U_orth)
            L_orth_i = torch.linalg.matrix_norm(J_D_orth, ord=2).item()
        else:
            L_orth_i = 0.0
        L_orth_list.append(L_orth_i)

    return {
        "delta": delta,
        "delta_rms": delta_rms,
        "delta_pixel": delta_pixel,
        "eps_iso_max": float(np.max(eps_iso_list)),
        "eps_iso_mean": float(np.mean(eps_iso_list)),
        "L_orth_max": float(np.max(L_orth_list)),
        "L_orth_mean": float(np.mean(L_orth_list))
    }


# ==========================================
# 3. Data Preparation & Subsampling (n=20)
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

mnist_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
full_loader = DataLoader(mnist_dataset, batch_size=60000, shuffle=False)
X_full, y_full = next(iter(full_loader))
X_full = X_full.to(device)
y_full = y_full.to(device)

N_full = X_full.shape[0]

torch.manual_seed(42)

num_classes = 10
n_per_class = N_SUBSAMPLE // num_classes

balanced_indices = []
for digit in range(num_classes):
    digit_indices = (y_full == digit).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(digit_indices))
    balanced_indices.append(digit_indices[perm[:n_per_class]])

balanced_indices = torch.cat(balanced_indices)
shuffle_perm = torch.randperm(len(balanced_indices))
final_indices = balanced_indices[shuffle_perm]

X_n = X_full[final_indices].clone().to(device)
y_n = y_full[final_indices].clone().to(device)

print(f"Loaded full dataset ({N_full} images). Subsampled n={N_SUBSAMPLE} for training.")


# ==========================================
# 4. Training strictly on X_n (n=20)
# ==========================================
encoder = GPEEncoder(AMBIENT_DIM, LATENT_DIM).to(device)
decoder = GPEDecoder(LATENT_DIM, AMBIENT_DIM).to(device)

# DataLoader containing ONLY the 20 subsampled images
train_subsample_dataset = TensorDataset(X_n)
train_loader = DataLoader(train_subsample_dataset, batch_size=N_SUBSAMPLE, shuffle=True)

if os.path.exists(ENC_CKPT):
    print(f"\n--- Loading pre-trained subsample models from checkpoint ---")
    encoder.load_state_dict(torch.load(ENC_CKPT, map_location=device))
    print(f"Loaded Encoder: {ENC_CKPT}")
if os.path.exists(DEC_CKPT):
    print(f"\n--- Loading pre-trained subsample models from checkpoint ---")
    decoder.load_state_dict(torch.load(DEC_CKPT, map_location=device))
    print(f"Loaded Decoder: {DEC_CKPT}")
    
# --- Stage 1: Encoder Training on n=20 ---
print(f"\n--- Stage 1: Training Encoder on n={N_SUBSAMPLE} ({EPOCHS_ENC} Epochs) ---")
optimizer_enc = optim.Adam(encoder.parameters(), lr=1e-3)
encoder.train()

for epoch in range(1, EPOCHS_ENC + 1):
    total_loss = 0.0
    for (x_batch,) in train_loader:
        x_batch = x_batch.to(device)
        optimizer_enc.zero_grad()

        z_batch = encoder(x_batch)

        # Restored raw distance computation (unnormalized)
        d2_x = torch.cdist(x_batch, x_batch, p=2)**2
        d2_z = torch.cdist(z_batch, z_batch, p=2)**2

        loss_enc = torch.mean((torch.log((1.0 + d2_z) / (1.0 + d2_x)))**2)
        loss_enc.backward()
        optimizer_enc.step()
        total_loss += loss_enc.item()

    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch:04d}/{EPOCHS_ENC} | Encoder Loss: {total_loss / len(train_loader):.6f}")

torch.save(encoder.state_dict(), ENC_CKPT)
print(f"Saved Encoder checkpoint to: {ENC_CKPT}")

# --- Stage 2: Decoder Training on n=20 ---
print(f"\n--- Stage 2: Training Decoder on n={N_SUBSAMPLE} ({EPOCHS_DEC} Epochs) ---")
optimizer_dec = optim.Adam(decoder.parameters(), lr=1e-3)
encoder.eval()
decoder.train()

for epoch in range(1, EPOCHS_DEC + 1):
    total_loss = 0.0
    for (x_batch,) in train_loader:
        x_batch = x_batch.to(device)
        optimizer_dec.zero_grad()

        with torch.no_grad():
            z_batch = encoder(x_batch)

        noise = torch.randn_like(z_batch) * SIGMA_TRAIN
        x_rec = decoder(z_batch + noise)

        loss_dec = torch.mean((x_rec - x_batch)**2)
        loss_dec.backward()
        optimizer_dec.step()
        total_loss += loss_dec.item()**0.5

    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch:04d}/{EPOCHS_DEC} | Decoder Rec Loss (square rooted): {total_loss / len(train_loader):.6f}")

torch.save(decoder.state_dict(), DEC_CKPT)
print(f"Saved Decoder checkpoint to: {DEC_CKPT}")


# ==========================================
# 5. Compute Empirical Latent Standard Deviation (sigma_z)
# ==========================================
encoder.eval()
decoder.eval()


def compute_sigma_z_median_pairwise(z_n):
    encoder.eval()
    with torch.no_grad():
        dists = torch.cdist(z_n, z_n, p=2)
        # Extract non-zero upper triangular distances
        mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
        sigma_z = torch.median(dists[mask]).item()
    print(f"Latent Scale (Median Pairwise Distance): {sigma_z:.5f}")
    return sigma_z

def compute_sigma_z_rms_norm(z_n):
    encoder.eval()
    with torch.no_grad():
        z_centered = z_n - z_n.mean(dim=0, keepdim=True)
        # RMS distance per dimension
        sigma_z = (torch.norm(z_centered, dim=1).mean() / np.sqrt(LATENT_DIM)).item()
    print(f"Latent Scale (RMS Norm per dim): {sigma_z:.5f}")
    return sigma_z

with torch.no_grad():
    z_all_sub = encoder(X_n)
    #sigma_z = torch.std(z_all_sub, dim=0).mean().item()
    sigma_z = compute_sigma_z_median_pairwise(z_all_sub)
    #sigma_z = compute_sigma_z_rms_norm(z_all_sub)

print(f"\nEmpirical Latent Standard Deviation (sigma_z): {sigma_z:.5f}")

print("\n--- Evaluating Theoretical Network Properties ---")
metrics = evaluate_theoretical_properties(
    encoder=encoder,
    decoder=decoder,
    X_sub=X_n,
    X_full=X_full,
    intrinsic_dim=10,
    eval_samples=N_SUBSAMPLE
)

print(f"Reconstruction Error (delta_rms):    {metrics['delta_rms']:.5f}")
print(f"Per-Pixel RMS Error (delta_pixel):   {metrics['delta_pixel']:.5f}")
print(f"Max Isometry Defect (eps_iso_max):   {metrics['eps_iso_max']:.5f}")
print(f"Mean Isometry Defect (eps_iso_mean): {metrics['eps_iso_mean']:.5f}")
print(f"Max Orthogonal Factor (L_orth_max):  {metrics['L_orth_max']:.5f}")
print(f"Mean Orthogonal Factor (L_orth_mean): {metrics['L_orth_mean']:.5f}")


# ==========================================
# 6. Save Image Reconstructions Grid vs. Sigma
# ==========================================
print("\n--- Generating Visual Comparison Grid of Smoothed Images ---")

SIGMA_VIS_LIST = [0.0, 0.1, 0.4, 0.8, 1.2]
n_vis_samples = 6
vis_samples = X_n[:n_vis_samples]

fig, axes = plt.subplots(
    nrows=len(SIGMA_VIS_LIST) * 2,
    ncols=n_vis_samples,
    figsize=(n_vis_samples * 1.5, len(SIGMA_VIS_LIST) * 3)
)

for row_idx, sig in enumerate(SIGMA_VIS_LIST):
    with torch.no_grad():
        amb_noise = torch.randn_like(vis_samples) * sig
        amb_img = torch.clamp(vis_samples + amb_noise, 0.0, 1.0).cpu().numpy()

        z_vis = encoder(vis_samples)
        # Scaling latent perturbations by sigma_z
        lat_noise = torch.randn_like(z_vis) * sig * sigma_z
        lat_img = decoder(z_vis + lat_noise).cpu().numpy()

    for col_idx in range(n_vis_samples):
        ax_amb = axes[row_idx * 2, col_idx]
        ax_amb.imshow(amb_img[col_idx].reshape(28, 28), cmap='gray')
        ax_amb.axis('off')
        if col_idx == 0:
            ax_amb.set_title(f"Ambient ($\sigma={sig}$)", fontsize=10, loc='left')

        ax_lat = axes[row_idx * 2 + 1, col_idx]
        ax_lat.imshow(lat_img[col_idx].reshape(28, 28), cmap='gray')
        ax_lat.axis('off')
        if col_idx == 0:
            ax_lat.set_title(f"Latent GPE ($\sigma={sig}$)", fontsize=10, loc='left')

plt.tight_layout()
grid_save_path = os.path.join(RESULTS_DIR, "smoothed_images_grid_subsample.png")
plt.savefig(grid_save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Image grid saved to: {grid_save_path}")


# ==========================================
# 7. Generate 11-Step Sigma Sweep Plot (Single Digit)
# ==========================================
print("\n--- Generating 11-Step Sigma Sweep Plot (0.0 to 1.0) ---")
for DIGIT in range(5):
    single_sample = X_n[DIGIT:DIGIT+1]
    SIGMA_SWEEP_8 = np.linspace(0.0, 2.0, 11)

    fig, axes = plt.subplots(nrows=2, ncols=11, figsize=(18, 4))

    with torch.no_grad():
        z_single = encoder(single_sample)

        for idx, sig in enumerate(SIGMA_SWEEP_8):
            amb_noise = torch.randn_like(single_sample) * sig
            amb_img = torch.clamp(single_sample + amb_noise, 0.0, 1.0).cpu().numpy().reshape(28, 28)

            # Scaling latent perturbations by sigma_z
            lat_noise = torch.randn_like(z_single) * sig * sigma_z
            lat_img = decoder(z_single + lat_noise).cpu().numpy().reshape(28, 28)

            ax_amb = axes[0, idx]
            ax_amb.imshow(amb_img, cmap='gray')
            ax_amb.set_title(f"$\sigma={sig:.2f}$", fontsize=11)
            ax_amb.axis('off')
            if idx == 0:
                ax_amb.text(-0.35, 0.5, "Ambient\nSmoothing", transform=ax_amb.transAxes,
                            fontsize=12, fontweight='bold', va='center', ha='right')

            ax_lat = axes[1, idx]
            ax_lat.imshow(lat_img, cmap='gray')
            ax_lat.axis('off')
            if idx == 0:
                ax_lat.text(-0.35, 0.5, "Latent GPE\nSmoothing", transform=ax_lat.transAxes,
                            fontsize=12, fontweight='bold', va='center', ha='right')

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    sweep_save_path = os.path.join(RESULTS_DIR, f"sigma_sweep_8steps_subsample_{DIGIT}.png")
    plt.savefig(sweep_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"11-step sigma sweep figure saved to: {sweep_save_path}")


# ==========================================
# 8. Evaluation: True W_2^2 Distance Sweep
# ==========================================
print("\n--- Evaluating W_2^2 Distances across Bandwidths ---")
ot_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, scaling=0.8)

SIGMA_GRID_NORM = np.linspace(0, 1.0, 50)
w2_ambient = []
w2_latent = []

eval_idx_ref = torch.randperm(N_full)[:N_EVAL]
X_ref = X_full[eval_idx_ref]

for sig in SIGMA_GRID_NORM:
    sub_idx = torch.randint(0, N_SUBSAMPLE, (N_EVAL,))
    X_sub = X_n[sub_idx]

    # 1. Ambient Smoothing
    noise_amb = torch.randn_like(X_sub) * sig
    X_amb_smoothed = torch.clamp(X_sub + noise_amb, 0.0, 1.0)

    # 2. Latent Smoothing
    with torch.no_grad():
        Z_sub = encoder(X_sub)
        # Scaling latent perturbations by sigma_z
        noise_lat = torch.randn_like(Z_sub) * sig * sigma_z
        X_lat_smoothed = decoder(Z_sub + noise_lat)

    loss_amb = 2.0 * ot_loss(X_amb_smoothed, X_ref).item()
    loss_lat = 2.0 * ot_loss(X_lat_smoothed, X_ref).item()

    w2_ambient.append(loss_amb)
    w2_latent.append(loss_lat)
    print(f"sigma = {sig:.3f} | Ambient W2^2: {loss_amb:.5f} | Latent W2^2: {loss_lat:.5f}")


# ==========================================
# 9. Save Plot of W_2^2 vs Sigma
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(SIGMA_GRID_NORM, w2_ambient, 'r-', label=f'Ambient Smoothing ($D={AMBIENT_DIM}$)', linewidth=2)
plt.plot(SIGMA_GRID_NORM, w2_latent, 'b-', label=f'Latent Smoothing GPE ($d={LATENT_DIM}$)', linewidth=2)

plt.xlabel(r'Bandwidth $\sigma$', fontsize=12)
plt.ylabel(r'$W_2^2(\cdot, \hat{\mu}_{\mathrm{full}})$', fontsize=12)
plt.title('MNIST Subsample (n=20 Trained): Ambient vs. Latent Smoothing', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

curve_save_path = os.path.join(RESULTS_DIR, "mnist_subsample_w2_comparison.png")
plt.savefig(curve_save_path, dpi=300)
plt.close()

print(f"\nExperiment complete.\nW2 comparison plot saved to: {curve_save_path}")
