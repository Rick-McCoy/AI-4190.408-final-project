# final_project_vae.py

# --- Imports ---
# Standard library
import argparse
import datetime
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml
from einops import rearrange, reduce, repeat
from sklearn.manifold import TSNE # For CPU-based t-SNE if needed for custom plots
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# --- Configuration Loading ---
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# --- Utility Functions ---
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
    elif torch.backends.mps.is_available(): # For Apple Silicon
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
        ]
    )
    # Suppress overly verbose Matplotlib font manager logs
    mpl_logger = logging.getLogger('matplotlib.font_manager')
    mpl_logger.setLevel(logging.WARNING)


# --- Dataset and DataLoader ---
class CelebADatasetWrapper(Dataset):
    def __init__(self, config: Dict[str, Any], split: str = 'train', target_attribute: Optional[str] = None):
        self.config = config
        self.image_size = config["dataset"]["image_size"]
        self.center_crop_size = config["dataset"]["center_crop_size"]
        self.root_dir = Path(config["dataset"]["root_dir"])
        self.target_attribute = target_attribute
        self.attributes_names = [] # Will be populated if target_attribute is used

        transform_list = [
            transforms.CenterCrop(self.center_crop_size),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to [-1, 1]
        ]
        self.transform = transforms.Compose(transform_list)

        target_type = ['attr'] if self.target_attribute else [] # Request attributes if needed

        # Check if CelebA dataset exists, if not, prompt to download
        # A bit tricky as torchvision downloads automatically if download=True
        # For simplicity, assume it will download or is present.

        self.celeba_dataset = datasets.CelebA(
            root=str(self.root_dir),
            split=split if config["dataset"]["celeba_use_predefined_splits"] else "all",
            target_type=target_type,
            transform=self.transform,
            download=True # Set to True to attempt download if not found
        )

        if self.target_attribute:
            self.attributes_names = self.celeba_dataset.attr_names
            try:
                self.target_attribute_idx = self.attributes_names.index(self.target_attribute)
            except ValueError:
                logging.error(f"Attribute '{self.target_attribute}' not found in CelebA attributes. Available: {self.attributes_names}")
                raise

        if not config["dataset"]["celeba_use_predefined_splits"] and split != "all":
            # Perform custom split
            total_size = len(self.celeba_dataset)
            train_ratio = config["dataset"]["split_ratios"]["train"]
            val_ratio = config["dataset"]["split_ratios"]["val"]

            train_size = int(train_ratio * total_size)
            val_size = int(val_ratio * total_size)
            test_size = total_size - train_size - val_size

            if train_size + val_size + test_size != total_size: # Check for rounding issues
                 # Adjust train_size to make sum exact, usually it's the largest
                train_size = total_size - val_size - test_size

            logging.info(f"Custom splitting 'all' data: Train={train_size}, Val={val_size}, Test={test_size}")

            # Ensure reproducible splits
            generator = torch.Generator().manual_seed(config["seed"])
            train_dataset, val_dataset, test_dataset = random_split(
                self.celeba_dataset, [train_size, val_size, test_size], generator=generator
            )

            if split == 'train':
                self.dataset_subset = train_dataset
            elif split == 'valid': # torchvision uses 'valid' for validation split name
                self.dataset_subset = val_dataset
            elif split == 'test':
                self.dataset_subset = test_dataset
            else:
                raise ValueError(f"Invalid split name for custom split: {split}")
        else:
            self.dataset_subset = self.celeba_dataset


    def __len__(self) -> int:
        return len(self.dataset_subset)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.target_attribute:
            img, attr = self.dataset_subset[idx]
            target_attr_val = attr[self.target_attribute_idx].float() # Ensure it's a float tensor
            return img, target_attr_val
        else:
            # If dataset_subset is from random_split, it directly returns the item.
            # If dataset_subset is CelebA itself, it might return (img, attr_tensor) or just img
            # depending on target_type. We only care about img here.
            item = self.dataset_subset[idx]
            if isinstance(item, tuple): # (img, attributes)
                return item[0]
            return item # just img

def get_dataloaders(config: Dict[str, Any], target_attribute_for_vis: Optional[str] = None) -> Dict[str, DataLoader]:
    dataloaders = {}
    splits_to_load = ['train', 'valid'] # Always load train and val
    if config["dataset"]["celeba_use_predefined_splits"] or \
       (1.0 - config["dataset"]["split_ratios"]["train"] - config["dataset"]["split_ratios"]["val"]) > 1e-5 : # if test split is non-zero
        splits_to_load.append('test')

    for split in splits_to_load:
        # For val/test loader used for embeddings, pass the target_attribute
        current_target_attribute = target_attribute_for_vis if split in ['valid', 'test'] else None
        dataset = CelebADatasetWrapper(config, split=split, target_attribute=current_target_attribute)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=config["dataset"]["batch_size"],
            shuffle=(split == 'train'),
            num_workers=config["dataset"]["num_workers"],
            pin_memory=config["dataset"]["pin_memory"],
            drop_last=(split == 'train') # Drop last incomplete batch for training
        )
        logging.info(f"Loaded {split} dataset with {len(dataset)} images.")
    return dataloaders

# --- Model: Convolutional VAE ---
class ConvolutionalVAE(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        model_config = config["model"]
        self.latent_dim = model_config["latent_dim"]
        encoder_channels = model_config["encoder_channels"] # e.g. [3, 32, 64, 128, 256]
        decoder_channels = model_config["decoder_channels"] # e.g. [256, 128, 64, 32, 3]
        kernel_size = model_config["kernel_size"]
        stride = model_config["stride"]
        padding = model_config["padding"]
        use_batch_norm = model_config["use_batch_norm"]
        activation_fn = nn.ReLU() if model_config["activation"] == "relu" else nn.LeakyReLU(0.2, inplace=True)

        # Encoder
        encoder_layers = []
        in_channels = encoder_channels[0]
        for out_channels in encoder_channels[1:]:
            encoder_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batch_norm)
            )
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(activation_fn)
            in_channels = out_channels
        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calculate flattened size after conv layers
        # Pass a dummy tensor through the encoder_conv to get the shape
        self._dummy_input_shape = (1, encoder_channels[0], config["dataset"]["image_size"], config["dataset"]["image_size"])
        dummy_input = torch.randn(self._dummy_input_shape)
        with torch.no_grad():
            self.flattened_size = self.encoder_conv(dummy_input).view(-1).shape[0]
            self._encoder_output_shape = self.encoder_conv(dummy_input).shape # For decoder unflatten

        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)

        # Decoder
        self.decoder_input_fc = nn.Linear(self.latent_dim, self.flattened_size)

        decoder_layers = []
        # Decoder input channels must match the flattened size channel dim
        in_channels = self._encoder_output_shape[1] # Last channel dim of encoder_conv output
        
        # The decoder_channels config should align with reversing the encoder.
        # Example: if encoder_conv output is 256 channels, decoder_channels[0] should be 256.
        # Decoder channels typically go from encoder's last conv output channel down to image channels.
        # So, decoder_channels should be something like [encoder_channels[-1], ..., encoder_channels[1], image_channels]
        # Let's adjust the interpretation:
        # decoder_channels = [latent_to_conv_channels, convT_hidden1, ..., image_channels]
        # For simplicity, use the provided decoder_channels directly, assuming they are set up to reverse encoder.
        
        current_channels = decoder_channels[0] # e.g. 256
        for i in range(len(decoder_channels) -1):
            out_channels = decoder_channels[i+1] # e.g. 128, then 64, etc.
            is_last_layer = (i == len(decoder_channels) - 2)
            
            decoder_layers.append(
                nn.ConvTranspose2d(current_channels, out_channels, kernel_size, stride, padding, bias=not use_batch_norm and not is_last_layer)
            )
            if not is_last_layer: # No BN or ReLU on the final output layer before Tanh
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm2d(out_channels))
                decoder_layers.append(activation_fn)
            current_channels = out_channels
        
        decoder_layers.append(nn.Tanh()) # Output images in [-1, 1]
        self.decoder_conv = nn.Sequential(*decoder_layers)


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h_flat = rearrange(h, 'b c h w -> b (c h w)') # Using einops
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_flat = self.decoder_input_fc(z)
        # Reshape h_flat to match the shape before flattening in encoder
        # Shape: (batch, channels, height, width)
        h = rearrange(h_flat, 'b (c h w) -> b c h w', 
                      c=self._encoder_output_shape[1], 
                      h=self._encoder_output_shape[2], 
                      w=self._encoder_output_shape[3])
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# --- Loss Function ---
def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reconstruction loss (Mean Squared Error for [-1,1] normalized images)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0] # Per-sample MSE
    # recon_loss = nn.functional.binary_cross_entropy_with_logits(recon_x_logits, (x + 1)/2, reduction='sum') / x.shape[0] # If outputting logits and using BCE

    # KL divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0] # Per-sample KLD

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
    # kl_beta: float
) -> Tuple[float, float, float]:
    model.train()
    total_loss_epoch = 0
    recon_loss_epoch = 0
    kld_loss_epoch = 0
    
    # kl_beta_current = kl_beta # Could implement annealing here
    kl_beta_current = 1.0 # Default for now, can be configured via training:kl_beta

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Train]", leave=False, file=sys.stdout)
    for batch_idx, data_batch in enumerate(progress_bar):
        if isinstance(data_batch, list): # (images, attributes)
            images = data_batch[0].to(device)
        else: # just images
            images = data_batch.to(device)

        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        
        loss, recon_loss, kld_loss = vae_loss_function(recon_images, images, mu, logvar, kl_beta_current)
        
        loss.backward()
        if config["training"]["gradient_clip_val"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_val"])
        optimizer.step()

        total_loss_epoch += loss.item()
        recon_loss_epoch += recon_loss.item()
        kld_loss_epoch += kld_loss.item()

        if batch_idx % config["logging"]["console_log_freq_metrics"] == 0:
            progress_bar.set_postfix({
                "Loss": loss.item() / images.size(0), # Per image
                "Recon": recon_loss.item() / images.size(0),
                "KLD": kld_loss.item() / images.size(0)
            })
        
        # TensorBoard batch-level logging (can be too much, usually epoch-level is fine)
        # global_step = epoch * len(dataloader) + batch_idx
        # writer.add_scalar("Loss/train_batch", loss.item(), global_step)
        # writer.add_scalar("ReconLoss/train_batch", recon_loss.item(), global_step)
        # writer.add_scalar("KLDLoss/train_batch", kld_loss.item(), global_step)

    avg_total_loss = total_loss_epoch / len(dataloader.dataset) # Per image
    avg_recon_loss = recon_loss_epoch / len(dataloader.dataset)
    avg_kld_loss = kld_loss_epoch / len(dataloader.dataset)
    
    return avg_total_loss, avg_recon_loss, avg_kld_loss

# --- Validation Step ---
def validate_epoch(
    model: ConvolutionalVAE,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    writer: SummaryWriter,
    # kl_beta: float
) -> Tuple[float, float, float]:
    model.eval()
    total_loss_epoch = 0
    recon_loss_epoch = 0
    kld_loss_epoch = 0
    # kl_beta_current = kl_beta
    kl_beta_current = 1.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']} [Val]", leave=False, file=sys.stdout)
    with torch.no_grad():
        for data_batch in progress_bar:
            if isinstance(data_batch, list): # (images, attributes)
                images = data_batch[0].to(device)
            else: # just images
                images = data_batch.to(device)

            recon_images, mu, logvar = model(images)
            loss, recon_loss, kld_loss = vae_loss_function(recon_images, images, mu, logvar, kl_beta_current)

            total_loss_epoch += loss.item()
            recon_loss_epoch += recon_loss.item()
            kld_loss_epoch += kld_loss.item()
            
            progress_bar.set_postfix({
                "Loss": loss.item() / images.size(0),
                "Recon": recon_loss.item() / images.size(0),
                "KLD": kld_loss.item() / images.size(0)
            })

    avg_total_loss = total_loss_epoch / len(dataloader.dataset)
    avg_recon_loss = recon_loss_epoch / len(dataloader.dataset)
    avg_kld_loss = kld_loss_epoch / len(dataloader.dataset)
    
    return avg_total_loss, avg_recon_loss, avg_kld_loss

# --- Visualization Functions ---
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    # Assuming normalization was to [-1, 1] using (0.5, 0.5, 0.5) mean and std
    return tensor * 0.5 + 0.5

def visualize_reconstructions(
    model: ConvolutionalVAE,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: Dict[str, Any],
    output_dir: Path
):
    model.eval()
    # Get a fixed batch for consistent visualization
    fixed_batch = next(iter(dataloader))
    if isinstance(fixed_batch, list):
        fixed_images = fixed_batch[0][:config["visualization"]["num_reconstructions"]].to(device)
    else:
        fixed_images = fixed_batch[:config["visualization"]["num_reconstructions"]].to(device)

    with torch.no_grad():
        recon_images, _, _ = model(fixed_images)

    # Concatenate original and reconstructed images
    comparison = torch.cat([fixed_images.cpu(), recon_images.cpu()])
    comparison_grid = make_grid(denormalize(comparison), nrow=fixed_images.size(0)) # Display side-by-side

    writer.add_image("Reconstructions/Original_vs_Reconstructed", comparison_grid, epoch)
    
    # Save to file for poster
    save_path = output_dir / f"reconstructions_epoch_{epoch+1:04d}.png"
    save_image(denormalize(comparison_grid), save_path)
    logging.info(f"Saved reconstructions to {save_path}")

def visualize_generated_samples(
    model: ConvolutionalVAE,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: Dict[str, Any],
    output_dir: Path
):
    model.eval()
    num_samples = config["visualization"]["num_generated_samples"]
    latent_dim = config["model"]["latent_dim"]
    
    # Generate from random latent vectors (prior)
    z_samples = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        generated_images = model.decode(z_samples)
    
    generated_grid = make_grid(denormalize(generated_images.cpu()), nrow=int(num_samples**0.5))
    writer.add_image("Generated/Samples_from_Prior", generated_grid, epoch)

    save_path = output_dir / f"generated_samples_epoch_{epoch+1:04d}.png"
    save_image(denormalize(generated_grid), save_path)
    logging.info(f"Saved generated samples to {save_path}")

def visualize_interpolations(
    model: ConvolutionalVAE,
    dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: Dict[str, Any],
    output_dir: Path
):
    model.eval()
    num_steps = config["visualization"]["num_interpolation_steps"]
    
    # Get two distinct images from the dataloader
    data_iter = iter(dataloader)
    batch1 = next(data_iter)
    if isinstance(batch1, list): img1 = batch1[0][0:1] 
    else: img1 = batch1[0:1]
    
    try: # Try to get a different image
        batch2 = next(data_iter)
        if isinstance(batch2, list): img2 = batch2[0][0:1]
        else: img2 = batch2[0:1]
        if torch.equal(img1, img2) and len(dataloader.dataset) > 1: # if same and more images exist
            batch2 = next(data_iter)
            if isinstance(batch2, list): img2 = batch2[0][0:1]
            else: img2 = batch2[0:1]
    except StopIteration: # If only one batch or one image
        if len(dataloader.dataset) > 1:
            if isinstance(batch1, list): img2 = batch1[0][1:2] # Use second image from the same batch
            else: img2 = batch1[1:2]
        else: # Only one image in dataset, interpolate with itself (less interesting)
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
    full_sequence = torch.cat([img1.cpu(), interpolated_images.cpu(), img2.cpu()], dim=0)
    interpolation_grid = make_grid(denormalize(full_sequence), nrow=num_steps + 2) # +2 for start/end
    
    writer.add_image("Generated/Interpolations", interpolation_grid, epoch)
    save_path = output_dir / f"interpolations_epoch_{epoch+1:04d}.png"
    save_image(denormalize(interpolation_grid), save_path)
    logging.info(f"Saved interpolations to {save_path}")

def log_latent_embeddings_to_tensorboard(
    model: ConvolutionalVAE,
    dataloader: DataLoader, # Should be val or test dataloader with attributes
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    config: Dict[str, Any]
):
    model.eval()
    num_embeddings = config["visualization"]["num_tsne_embeddings"]
    target_attribute = config["visualization"]["tsne_target_attribute"]

    if not target_attribute:
        logging.info("Skipping latent embeddings visualization as 'tsne_target_attribute' is not set.")
        return

    all_latents = []
    all_labels = [] # For attribute values
    all_images = []

    count = 0
    with torch.no_grad():
        for data_batch in dataloader:
            if count >= num_embeddings:
                break
            
            images, attributes = data_batch # Assumes dataloader yields (img, attr)
            images = images.to(device)
            attributes = attributes.to(device) # Target attribute value

            mu, _ = model.encode(images) # Use mu as the representation
            
            num_to_take = min(images.size(0), num_embeddings - count)
            
            all_latents.append(mu[:num_to_take].cpu())
            all_labels.append(attributes[:num_to_take].cpu())
            all_images.append(denormalize(images[:num_to_take].cpu())) # Store images for TensorBoard projector
            
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
        metadata=label_names, # List of strings
        label_img=images_tensor, # (N, C, H, W)
        global_step=epoch,
        tag=f"Latent_Space_{target_attribute}"
    )
    logging.info(f"Logged {count} latent embeddings to TensorBoard projector for epoch {epoch+1}.")


# --- Main Function ---
def main(config_path: Union[str, Path]):
    config = load_config(config_path)
    
    # Setup output directory
    experiment_dir = Path(config["output_base_dir"]) / config["experiment_name"] / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    dataloaders = get_dataloaders(config, target_attribute_for_vis=config["visualization"]["tsne_target_attribute"])
    train_loader = dataloaders['train']
    val_loader = dataloaders['valid']
    # test_loader = dataloaders.get('test', None) # Optional

    # Model
    model = ConvolutionalVAE(config).to(device)
    logging.info(f"Model:\n{model}")
    
    # Log model graph to TensorBoard (requires a dummy input)
    # Ensure dummy input matches model's expected input (after transforms)
    dummy_input_tb = torch.randn(
        config["dataset"]["batch_size"],
        config["model"]["encoder_channels"][0], # num_input_channels
        config["dataset"]["image_size"],
        config["dataset"]["image_size"]
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
            weight_decay=config["training"]["weight_decay"]
        )
    else: # Add other optimizers if needed
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    logging.info(f"Optimizer: {optimizer}")

    # Scheduler (optional)
    scheduler = None
    if config["training"].get("scheduler") and config["training"]["scheduler"].get("name"):
        scheduler_name = config["training"]["scheduler"]["name"].lower()
        if scheduler_name == "reducelronplateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min', # on validation loss
                factor=config["training"]["scheduler"]["factor"],
                patience=config["training"]["scheduler"]["patience"],
                verbose=True
            )
        elif scheduler_name == "steplr":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["training"]["scheduler"]["step_size"],
                gamma=config["training"]["scheduler"]["gamma"]
            )
        else:
            logging.warning(f"Unsupported scheduler: {scheduler_name}. No scheduler will be used.")
        if scheduler:
            logging.info(f"Scheduler: {scheduler_name} configured.")


    # --- Training Loop ---
    best_val_loss = float('inf')
    saved_models_dir = experiment_dir / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir = experiment_dir / "visualizations"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    # KL Annealing (simple linear, if configured)
    # kl_anneal_epochs = config["training"].get("kl_anneal_epochs", 0)
    # kl_beta_config = config["training"].get("kl_beta", 1.0)

    logging.info("Starting training...")
    for epoch in range(config["training"]["epochs"]):
        # current_kl_beta = kl_beta_config
        # if kl_anneal_epochs > 0 and epoch < kl_anneal_epochs:
        #     current_kl_beta = kl_beta_config * (epoch / kl_anneal_epochs)
        
        train_loss, train_recon, train_kld = train_epoch(
            model, train_loader, optimizer, device, epoch, config, writer #, current_kl_beta
        )
        val_loss, val_recon, val_kld = validate_epoch(
            model, val_loader, device, epoch, config, writer #, current_kl_beta
        )

        print(f"Epoch {epoch+1}/{config['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KLD: {train_kld:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KLD: {val_kld:.4f})")
        logging.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} (R={train_recon:.4f}, KLD={train_kld:.4f}), "
                     f"Val Loss={val_loss:.4f} (R={val_recon:.4f}, KLD={val_kld:.4f})")

        # TensorBoard scalars
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("ReconLoss/train_epoch", train_recon, epoch)
        writer.add_scalar("KLDLoss/train_epoch", train_kld, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("ReconLoss/val_epoch", val_recon, epoch)
        writer.add_scalar("KLDLoss/val_epoch", val_kld, epoch)
        # writer.add_scalar("KL_Beta", current_kl_beta, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

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
        if (epoch + 1) % config["logging"]["tensorboard_log_freq_images"] == 0 or epoch == config["training"]["epochs"] -1 :
            logging.info(f"Generating visualizations for epoch {epoch+1}...")
            visualize_reconstructions(model, val_loader, device, writer, epoch, config, visualizations_dir)
            visualize_generated_samples(model, device, writer, epoch, config, visualizations_dir)
            visualize_interpolations(model, val_loader, device, writer, epoch, config, visualizations_dir)
            if config["visualization"]["tsne_target_attribute"]:
                 # Ensure val_loader for embeddings has attributes
                val_loader_for_embeddings = DataLoader(
                    CelebADatasetWrapper(config, split='valid', target_attribute=config["visualization"]["tsne_target_attribute"]),
                    batch_size=config["dataset"]["batch_size"], shuffle=False, # No shuffle for consistent embeddings
                    num_workers=config["dataset"]["num_workers"], pin_memory=config["dataset"]["pin_memory"]
                )
                log_latent_embeddings_to_tensorboard(model, val_loader_for_embeddings, device, writer, epoch, config)

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
        default="config.yaml", # Expect config.yaml in the same directory
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.is_file():
        print(f"Error: Config file not found at {config_file}")
        # Create a default config if it doesn't exist, for convenience
        default_config_content = """
experiment_name: "celeba_vae_default"
output_base_dir: "runs_vae"

dataset:
  name: "CelebA"
  root_dir: "./data"
  image_size: 64
  center_crop_size: 140
  batch_size: 128 # Adjusted for typical GPU memory
  num_workers: 4
  pin_memory: true
  split_ratios:
    train: 0.8
    val: 0.1
  celeba_use_predefined_splits: true

model:
  type: "ConvolutionalVAE"
  latent_dim: 128
  encoder_channels: [3, 32, 64, 128, 256] # For 64x64 input
  decoder_channels: [256, 128, 64, 32, 3] # For 64x64 output
  kernel_size: 4
  stride: 2
  padding: 1
  use_batch_norm: true
  activation: "relu"

training:
  epochs: 25 # Reduced for a quick test
  learning_rate: 0.0002
  optimizer: "Adam"
  adam_betas: [0.5, 0.999]
  weight_decay: 0.0
  # kl_beta: 1.0
  # kl_anneal_epochs: 0
  scheduler:
    name: null
  gradient_clip_val: null

logging:
  log_level: "INFO"
  tensorboard_log_freq_images: 5
  console_log_freq_metrics: 50

visualization:
  num_reconstructions: 16
  num_generated_samples: 16
  num_interpolation_steps: 8
  num_tsne_embeddings: 500 # Reduced for speed
  tsne_target_attribute: "Smiling" # "Male", "Eyeglasses", null

seed: 42
        """
        with open("config.yaml", "w") as f_default:
            f_default.write(default_config_content)
        print("Created a default config.yaml. Please review and run again.")
        sys.exit(1)
        
    main(config_file)
