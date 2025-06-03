# --- Configuration Loading ---
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from model import ConvolutionalVAE


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN if used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_logging(log_path: Path, log_level: str = "INFO") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            # No console handler for debug, as requested.
            # TQDM will handle console progress.
            # High-level messages can be printed directly or logged with INFO+.
        ],
    )
    # Suppress overly verbose Matplotlib font manager logs
    mpl_logger = logging.getLogger("matplotlib.font_manager")
    mpl_logger.setLevel(logging.WARNING)


def denormalize(tensor: Tensor) -> Tensor:
    # Assuming normalization was to [-1, 1] using (0.5, 0.5, 0.5) mean and std
    return tensor * 0.5 + 0.5


def visualize_reconstructions(
    model: ConvolutionalVAE,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: dict[str, Any],
    output_dir: Path,
):
    model.eval()
    # Get a fixed batch for consistent visualization
    fixed_batch = next(iter(dataloader))
    if isinstance(fixed_batch, list):
        fixed_images = fixed_batch[0][
            : config["visualization"]["num_reconstructions"]
        ].to(device)
    else:
        fixed_images = fixed_batch[: config["visualization"]["num_reconstructions"]].to(
            device
        )

    with torch.no_grad():
        recon_images, _, _ = model(fixed_images)

    # Concatenate original and reconstructed images
    comparison = torch.cat([fixed_images.cpu(), recon_images.cpu()])
    comparison_grid = make_grid(
        denormalize(comparison), nrow=fixed_images.size(0)
    )  # Display side-by-side

    writer.add_image(
        "Reconstructions/Original_vs_Reconstructed", comparison_grid, epoch
    )

    # Save to file for poster
    save_path = output_dir / f"reconstructions_epoch_{epoch + 1:04d}.png"
    save_image(comparison_grid, save_path)
    logging.info(f"Saved reconstructions to {save_path}")


def visualize_generated_samples(
    model: ConvolutionalVAE,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: dict[str, Any],
    output_dir: Path,
):
    model.eval()
    num_samples = config["visualization"]["num_generated_samples"]
    latent_dim = config["model"]["latent_dim"]

    # Generate from random latent vectors (prior)
    z_samples = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        generated_images = model.decode(z_samples)

    generated_grid = make_grid(
        denormalize(generated_images.cpu()), nrow=int(num_samples**0.5)
    )
    writer.add_image("Generated/Samples_from_Prior", generated_grid, epoch)

    save_path = output_dir / f"generated_samples_epoch_{epoch + 1:04d}.png"
    save_image(generated_grid, save_path)
    logging.info(f"Saved generated samples to {save_path}")


def visualize_interpolations(
    model: ConvolutionalVAE,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: dict[str, Any],
    output_dir: Path,
):
    model.eval()
    num_steps = config["visualization"]["num_interpolation_steps"]

    # Get two distinct images from the dataloader
    data_iter = iter(dataloader)
    batch1 = next(data_iter)
    if isinstance(batch1, list):
        img1 = batch1[0][0:1]
    else:
        img1 = batch1[0:1]

    try:  # Try to get a different image
        batch2 = next(data_iter)
        if isinstance(batch2, list):
            img2 = batch2[0][0:1]
        else:
            img2 = batch2[0:1]
        if (
            torch.equal(img1, img2) and len(dataloader.dataset) > 1
        ):  # if same and more images exist
            batch2 = next(data_iter)
            if isinstance(batch2, list):
                img2 = batch2[0][0:1]
            else:
                img2 = batch2[0:1]
    except StopIteration:  # If only one batch or one image
        if len(dataloader.dataset) > 1:
            if isinstance(batch1, list):
                img2 = batch1[0][1:2]  # Use second image from the same batch
            else:
                img2 = batch1[1:2]
        else:  # Only one image in dataset, interpolate with itself (less interesting)
            img2 = img1.clone()

    img1, img2 = img1.to(device), img2.to(device)

    with torch.no_grad():
        mu1, logvar1 = model.encode(img1)
        mu2, logvar2 = model.encode(img2)

        # For simplicity, interpolate mu. Can also use reparameterized z.
        z1 = model.reparameterize(mu1, logvar1)
        z2 = model.reparameterize(mu2, logvar2)

        interpolated_latents = []
        for alpha in np.linspace(0, 1, num_steps):
            # Linear interpolation (lerp)
            z_interp = (1 - alpha) * z1 + alpha * z2
            # Spherical linear interpolation (slerp) - more complex, lerp is fine for VAEs usually
            # omega = torch.acos(torch.clamp(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)), -1, 1))
            # if torch.abs(omega) < 1e-4: # if vectors are too close
            #     z_interp = z1
            # else:
            #     z_interp = (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * z1 + \
            #                (torch.sin(alpha * omega) / torch.sin(omega)) * z2
            interpolated_latents.append(z_interp)

        z_interp_all = torch.cat(interpolated_latents, dim=0)
        interpolated_images = model.decode(z_interp_all)

    # Include start and end images in the grid
    full_sequence = torch.cat(
        [img1.cpu(), interpolated_images.cpu(), img2.cpu()], dim=0
    )
    interpolation_grid = make_grid(
        denormalize(full_sequence), nrow=num_steps + 2
    )  # +2 for start/end

    writer.add_image("Generated/Interpolations", interpolation_grid, epoch)
    save_path = output_dir / f"interpolations_epoch_{epoch + 1:04d}.png"
    save_image(interpolation_grid, save_path)
    logging.info(f"Saved interpolations to {save_path}")


def log_latent_embeddings_to_tensorboard(
    model: ConvolutionalVAE,
    dataloader: DataLoader,  # Should be val or test dataloader with attributes
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: dict[str, Any],
):
    model.eval()
    num_embeddings = config["visualization"]["num_tsne_embeddings"]
    target_attribute = config["visualization"]["tsne_target_attribute"]

    if not target_attribute:
        logging.info(
            "Skipping latent embeddings visualization as 'tsne_target_attribute' is not set."
        )
        return

    all_latents = []
    all_labels = []  # For attribute values
    all_images = []

    count = 0
    with torch.no_grad():
        for data_batch in dataloader:
            if count >= num_embeddings:
                break

            images, attributes = data_batch  # Assumes dataloader yields (img, attr)
            images = images.to(device)
            attributes = attributes.to(device)  # Target attribute value

            mu, _ = model.encode(images)  # Use mu as the representation

            num_to_take = min(images.size(0), num_embeddings - count)

            all_latents.append(mu[:num_to_take].cpu())
            all_labels.append(attributes[:num_to_take].cpu())
            all_images.append(
                denormalize(images[:num_to_take].cpu())
            )  # Store images for TensorBoard projector

            count += num_to_take

    if not all_latents:
        logging.warning("No data collected for latent embeddings.")
        return

    latents_tensor = torch.cat(all_latents, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    images_tensor = torch.cat(all_images, dim=0)

    # Ensure labels are suitable for TensorBoard (e.g., class indices or string names)
    # For binary attributes (0 or 1), it's fine.
    # For continuous, might need binning or just use values.
    # Here, CelebA attributes are 0 or 1. Let's map to strings for clarity.
    label_names = [f"{target_attribute}: {int(l.item())}" for l in labels_tensor]

    writer.add_embedding(
        mat=latents_tensor,
        metadata=label_names,  # List of strings
        label_img=images_tensor,  # (N, C, H, W)
        global_step=epoch,
        tag=f"Latent_Space_{target_attribute}",
    )
    logging.info(
        f"Logged {count} latent embeddings to TensorBoard projector for epoch {epoch + 1}."
    )
