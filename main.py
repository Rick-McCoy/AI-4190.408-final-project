# final_project_vae.py

# --- Imports ---
# Standard library
import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import yaml

# Third-party
from einops import reduce
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import CelebADatasetWrapper, get_dataloaders
from model import ConvolutionalVAE
from utils import (
    get_device,
    load_config,
    log_latent_embeddings_to_tensorboard,
    set_seed,
    setup_logging,
    visualize_generated_samples,
    visualize_interpolations,
    visualize_reconstructions,
)


# --- Loss Function ---
def vae_loss_function(
    recon_x: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    kl_beta: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    recon_loss = F.mse_loss(recon_x, x, reduction="none")
    recon_loss = reduce(recon_loss, "b c h w -> b", "sum").mean()

    # KL divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = reduce(kld_loss, "b l -> b", "sum").mean()

    total_loss = recon_loss + kl_beta * kld_loss
    return total_loss, recon_loss, kld_loss


# --- Training Step ---
def train_epoch(
    model: ConvolutionalVAE,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    writer: SummaryWriter,
    kl_beta: float,
):
    model.train()

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{config['training']['epochs']} [Train]",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for batch_idx, data_batch in enumerate(progress_bar):
        if isinstance(data_batch, list):  # (images, attributes)
            images = data_batch[0].to(device)
        else:  # just images
            images = data_batch.to(device)

        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)

        loss, recon_loss, kld_loss = vae_loss_function(
            recon_images, images, mu, logvar, kl_beta
        )

        loss.backward()
        if config["training"]["gradient_clip_val"] is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["gradient_clip_val"]
            )
        optimizer.step()

        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("Train/loss", loss, global_step=global_step)
        writer.add_scalar("Train/recon_loss", recon_loss, global_step=global_step)
        writer.add_scalar("Train/kld_loss", kld_loss, global_step=global_step)

        if batch_idx % config["logging"]["console_log_freq_metrics"] == 0:
            progress_bar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Recon": recon_loss.item(),
                    "KLD": kld_loss.item(),
                }
            )


# --- Validation Step ---
def validate_epoch(
    model: ConvolutionalVAE,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    writer: SummaryWriter,
    kl_beta: float,
):
    model.eval()
    loss_list = []
    recon_loss_list = []
    kld_loss_list = []

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{config['training']['epochs']} [Val]",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for data_batch in progress_bar:
            if isinstance(data_batch, list):  # (images, attributes)
                images = data_batch[0].to(device)
            else:  # just images
                images = data_batch.to(device)

            recon_images, mu, logvar = model(images)
            loss, recon_loss, kld_loss = vae_loss_function(
                recon_images, images, mu, logvar, kl_beta
            )

            loss_list.append(loss.item())
            recon_loss_list.append(recon_loss.item())
            kld_loss_list.append(kld_loss.item())

            progress_bar.set_postfix(
                {
                    "Loss": loss.item(),
                    "Recon": recon_loss.item(),
                    "KLD": kld_loss.item(),
                }
            )

    avg_total_loss = sum(loss_list) / len(loss_list)
    avg_recon_loss = sum(recon_loss_list) / len(recon_loss_list)
    avg_kld_loss = sum(kld_loss_list) / len(kld_loss_list)

    writer.add_scalar("Val/loss", avg_total_loss, epoch)
    writer.add_scalar("Val/recon_loss", avg_recon_loss, epoch)
    writer.add_scalar("Val/kld_loss", avg_kld_loss, epoch)

    return avg_total_loss


# --- Main Function ---
def main(config_path: Union[str, Path]):
    config = load_config(config_path)

    # Setup output directory
    experiment_dir = (
        Path(config["output_base_dir"])
        / config["experiment_name"]
        / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config to experiment dir
    with open(experiment_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Setup logging
    log_file_path = experiment_dir / "training.log"
    setup_logging(log_file_path, config["logging"]["log_level"])
    logging.info(f"Configuration loaded from: {config_path}")
    logging.info(f"Experiment outputs will be saved to: {experiment_dir}")

    # Setup reproducibility and device
    set_seed(config["seed"])
    device = get_device()
    logging.info(f"Using device: {device}")

    # Setup TensorBoard writer
    tensorboard_dir = experiment_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logging.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")

    # Dataloaders
    dataloaders = get_dataloaders(
        config,
        target_attribute_for_vis=config["visualization"]["tsne_target_attribute"],
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["valid"]
    # test_loader = dataloaders.get('test', None) # Optional

    # Model
    model = ConvolutionalVAE(config).to(device)
    logging.info(f"Model:\n{model}")

    # Log model graph to TensorBoard (requires a dummy input)
    # Ensure dummy input matches model's expected input (after transforms)
    dummy_input_tb = torch.randn(
        config["dataset"]["batch_size"],
        config["model"]["encoder_channels"][0],  # num_input_channels
        config["dataset"]["image_size"],
        config["dataset"]["image_size"],
    ).to(device)
    try:
        writer.add_graph(model, dummy_input_tb)
    except Exception as e:
        logging.warning(f"Could not add model graph to TensorBoard: {e}")

    # Optimizer
    if config["training"]["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            betas=tuple(config["training"]["adam_betas"]),
            weight_decay=config["training"]["weight_decay"],
        )
    else:  # Add other optimizers if needed
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    logging.info(f"Optimizer: {optimizer}")

    # Scheduler (optional)
    scheduler = None
    if config["training"].get("scheduler") and config["training"]["scheduler"].get(
        "name"
    ):
        scheduler_name = config["training"]["scheduler"]["name"].lower()
        if scheduler_name == "reducelronplateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",  # on validation loss
                factor=config["training"]["scheduler"]["factor"],
                patience=config["training"]["scheduler"]["patience"],
                # verbose=True
            )
        elif scheduler_name == "steplr":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["training"]["scheduler"]["step_size"],
                gamma=config["training"]["scheduler"]["gamma"],
            )
        else:
            logging.warning(
                f"Unsupported scheduler: {scheduler_name}. No scheduler will be used."
            )
        if scheduler:
            logging.info(f"Scheduler: {scheduler_name} configured.")

    # --- Training Loop ---
    best_val_loss = float("inf")
    saved_models_dir = experiment_dir / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir = experiment_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    # KL Annealing (simple linear, if configured)
    kl_anneal_epochs = config["training"].get("kl_anneal_epochs", 0)
    kl_beta_config = config["training"].get("kl_beta", 1.0)

    logging.info("Starting training...")
    for epoch in range(config["training"]["epochs"]):
        current_kl_beta = kl_beta_config
        if kl_anneal_epochs > 0 and epoch < kl_anneal_epochs:
            current_kl_beta = kl_beta_config * (epoch / kl_anneal_epochs)

        train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            config,
            writer,
            current_kl_beta,
        )
        val_loss = validate_epoch(
            model, val_loader, device, epoch, config, writer, current_kl_beta
        )

        # TensorBoard scalars
        writer.add_scalar("KL_Beta", current_kl_beta, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save model checkpoint (best and latest)
        torch.save(model.state_dict(), saved_models_dir / "latest_model.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), saved_models_dir / "best_model.pth")
            logging.info(f"Saved new best model with val_loss: {best_val_loss:.4f}")

        # Visualizations (periodically)
        if (epoch + 1) % config["logging"][
            "tensorboard_log_freq_images"
        ] == 0 or epoch == config["training"]["epochs"] - 1:
            logging.info(f"Generating visualizations for epoch {epoch + 1}...")
            visualize_reconstructions(
                model, val_loader, device, writer, epoch, config, visualizations_dir
            )
            visualize_generated_samples(
                model, device, writer, epoch, config, visualizations_dir
            )
            visualize_interpolations(
                model, val_loader, device, writer, epoch, config, visualizations_dir
            )
            if config["visualization"]["tsne_target_attribute"]:
                # Ensure val_loader for embeddings has attributes
                val_loader_for_embeddings = DataLoader(
                    CelebADatasetWrapper(
                        config,
                        split="valid",
                        target_attribute=config["visualization"][
                            "tsne_target_attribute"
                        ],
                    ),
                    batch_size=config["dataset"]["batch_size"],
                    shuffle=False,  # No shuffle for consistent embeddings
                    num_workers=config["dataset"]["num_workers"],
                    pin_memory=config["dataset"]["pin_memory"],
                )
                log_latent_embeddings_to_tensorboard(
                    model, val_loader_for_embeddings, device, writer, epoch, config
                )

    writer.close()
    logging.info("Training finished.")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Models saved in: {saved_models_dir}")
    logging.info(f"Visualizations saved in: {visualizations_dir}")
    logging.info(f"TensorBoard logs in: {tensorboard_dir}")
    print(f"Training complete. Outputs in {experiment_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolutional VAE Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",  # Expect config.yaml in the same directory
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.is_file():
        raise ValueError(f"No valid configuration file at {args.config}")

    main(config_file)
