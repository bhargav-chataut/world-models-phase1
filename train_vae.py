"""
train_vae.py
Training loop for V (the ConvVAE from vae.py). Meant to be driven from a
thin Colab notebook, e.g.:

    !git clone https://github.com/YOUR_USERNAME/world-models-phase1.git
    %cd world-models
    from train_vae import train

    train(
        memmap_path="/content/drive/MyDrive/world-models/frames_part",
        checkpoint_dir="/content/drive/MyDrive/world-models/checkpoints",
        epochs=20,
    )

Checkpoints and a reconstruction sample plot are saved every epoch to
Drive, so a disconnected Colab runtime never loses progress -- resume
from the last checkpoint instead of starting over.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import matplotlib.pyplot as plt

from vae import ConvVAE, vae_loss
from data_loader import FrameDataset


def kl_warmup_weight(epoch, warmup_epochs=10, target=1.0):
    """Ramp KL weight from 0 -> target over the first `warmup_epochs`.
    Prevents the classic VAE failure mode where KL dominates early and
    reconstructions collapse to a blurry average frame before the model
    has learned anything useful to encode."""
    if epoch >= warmup_epochs:
        return target
    return target * (epoch / warmup_epochs)


def save_reconstruction_sample(model, batch, epoch, out_dir, device):
    """Saves a side-by-side original vs. reconstruction grid so you can
    visually track training progress, not just watch the loss number."""
    model.eval()
    with torch.no_grad():
        x = batch[:8].to(device)
        recon, _, _ = model(x)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        orig = x[i].cpu().permute(1, 2, 0).numpy()
        rec = recon[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("orig", fontsize=10)
    axes[1, 0].set_ylabel("recon", fontsize=10)
    fig.suptitle(f"Epoch {epoch}")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"recon_epoch{epoch:03d}.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)
    model.train()
    return path


def train(
    memmap_path,
    checkpoint_dir,
    epochs=20,
    batch_size=128,
    lr=1e-4,
    latent_dim=32,
    kl_warmup_epochs=10,
    kl_target_weight=1.0,
    val_fraction=0.05,
    resume=True,
    num_workers=2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- check Colab Runtime > Change "
              "runtime type. Training on CPU will be extremely slow.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    recon_dir = os.path.join(checkpoint_dir, "reconstructions")

    # --- data ---
    full_ds = FrameDataset(memmap_path)
    val_size = int(len(full_ds) * val_fraction)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"Train frames: {train_size} | Val frames: {val_size}")

    # --- model ---
    model = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    start_epoch = 0
    ckpt_path = os.path.join(checkpoint_dir, "vae_latest.pt")
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']} "
              f"-> continuing at epoch {start_epoch}")

    # Grab one fixed validation batch up front so reconstruction samples
    # are visually comparable across epochs (same input frames each time).
    fixed_val_batch = next(iter(val_loader))

    history = {"train_loss": [], "val_loss": [], "train_recon": [], "train_kld": []}

    for epoch in range(start_epoch, epochs):
        model.train()
        kl_w = kl_warmup_weight(epoch, kl_warmup_epochs, kl_target_weight)

        running_loss, running_recon, running_kld, n_batches = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kld = vae_loss(recon, x, mu, logvar, kld_weight=kl_w)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kld += kld.item()
            n_batches += 1

        train_loss = running_loss / n_batches
        train_recon = running_recon / n_batches
        train_kld = running_kld / n_batches

        # --- validation ---
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar, kld_weight=kl_w)
                val_running += loss.item()
        val_loss = val_running / len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_recon"].append(train_recon)
        history["train_kld"].append(train_kld)

        print(f"Epoch {epoch:03d} | kl_w={kl_w:.3f} | "
              f"train_loss={train_loss:.2f} (recon={train_recon:.2f}, kld={train_kld:.2f}) | "
              f"val_loss={val_loss:.2f}")

        # --- save checkpoint + reconstruction sample every epoch ---
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "latent_dim": latent_dim,
        }, ckpt_path)

        recon_path = save_reconstruction_sample(model, fixed_val_batch, epoch, recon_dir, device)
        if epoch == start_epoch:
            print(f"Reconstruction samples saving to: {recon_dir}")

    print(f"Training done. Final checkpoint: {ckpt_path}")
    return model, history


if __name__ == "__main__":
    pass