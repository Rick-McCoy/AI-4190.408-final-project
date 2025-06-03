# --- Residual Block (no spatial dimension change) ---
import torch
from einops import rearrange
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool,
        activation_str: str,
    ):
        super().__init__()

        # Main path
        self.bn1 = nn.BatchNorm2d(in_channels) if use_batch_norm else nn.Identity()
        self.act1 = (
            nn.ReLU(inplace=True)
            if activation_str == "relu"
            else nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,  # Standard 3x3
            stride=1,  # No spatial change
            padding=1,  # Keep dimensions same
            bias=not use_batch_norm,
        )

        self.bn2 = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.act2 = (
            nn.ReLU(inplace=True)
            if activation_str == "relu"
            else nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Conv2d(
            out_channels,  # Input to second conv is output of first
            out_channels,  # Output of second conv is same as its input
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_batch_norm,
        )

        # Shortcut / Identity path
        if in_channels != out_channels:  # Only project if channels change
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,  # 1x1 conv for projection
                stride=1,  # No spatial change in shortcut
                padding=0,
                bias=False,  # Often no bias if BN is used
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        out = self.bn1(x)
        out = self.act1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = out + identity
        return out


# --- Model: Convolutional VAE with Residual Blocks and Pooling/Upsampling ---
class ConvolutionalVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_config = config["model"]
        self.latent_dim = model_config["latent_dim"]

        encoder_channels = model_config[
            "encoder_channels"
        ]  # e.g. [3, 32, 64, 128, 256]
        decoder_channels = model_config[
            "decoder_channels"
        ]  # e.g. [256, 128, 64, 32, 3]

        use_batch_norm = model_config["use_batch_norm"]
        activation_str = model_config.get("activation", "relu")

        image_size = config["dataset"]["image_size"]
        # Target spatial dimension of the feature map before flattening
        final_conv_spatial_dim = model_config.get("final_conv_spatial_dim", 4)
        pooling_type = model_config.get("pooling_type", "max")  # "max" or "avg"
        upsample_mode = model_config.get(
            "upsample_mode", "bilinear"
        )  # "nearest", "linear", "bilinear", "bicubic", "trilinear"

        # --- Determine number of downsampling/upsampling stages ---
        self.num_spatial_stages = 0
        current_dim = image_size
        if final_conv_spatial_dim <= 0:
            final_conv_spatial_dim = 1  # Ensure positive
        while current_dim > final_conv_spatial_dim:
            if current_dim % 2 != 0:
                raise ValueError(
                    f"Image size {current_dim} not cleanly divisible by 2 for pooling."
                )
            current_dim //= 2
            self.num_spatial_stages += 1

        if len(encoder_channels) - 1 < self.num_spatial_stages:
            raise ValueError(
                f"Encoder channels list length ({len(encoder_channels) - 1}) is less than "
                f"required num_spatial_stages ({self.num_spatial_stages})"
            )
        if len(decoder_channels) - 1 < self.num_spatial_stages:
            raise ValueError(
                f"Decoder channels list length ({len(decoder_channels) - 1} for ResBlocks) is less than "
                f"required num_spatial_stages ({self.num_spatial_stages})"
            )

        # --- Encoder ---
        encoder_blocks_list = []
        in_c = encoder_channels[0]
        num_encoder_resblocks = len(encoder_channels) - 1

        for i in range(num_encoder_resblocks):
            out_c: int = encoder_channels[i + 1]
            encoder_blocks_list.append(
                ResidualBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    use_batch_norm=use_batch_norm,
                    activation_str=activation_str,
                )
            )
            # Add pooling layer after the ResBlock if it's one of the downsampling stages
            if i < self.num_spatial_stages:
                if pooling_type == "max":
                    encoder_blocks_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                elif pooling_type == "avg":
                    encoder_blocks_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
                else:
                    raise ValueError(f"Unsupported pooling_type: {pooling_type}")
            in_c = out_c
        self.encoder_conv = nn.Sequential(*encoder_blocks_list)

        # Calculate flattened size after conv layers
        self._dummy_input_shape = (
            1,  # batch size
            encoder_channels[0],
            image_size,
            image_size,
        )
        dummy_input = torch.randn(self._dummy_input_shape)
        with torch.no_grad():
            encoded_features = self.encoder_conv(dummy_input)
            self.flattened_size = encoded_features.view(-1).shape[0]
            self._encoder_output_shape = encoded_features.shape  # For decoder unflatten

        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)

        # --- Decoder ---
        self.decoder_input_fc = nn.Linear(self.latent_dim, self.flattened_size)
        decoder_blocks_list = []

        num_decoder_resblocks_stages = len(decoder_channels) - 1

        for i in range(num_decoder_resblocks_stages):
            in_c = decoder_channels[i]
            out_c = decoder_channels[i + 1]
            # Add Upsampling layer before the ResBlock if it's one of the upsampling stages
            if i < self.num_spatial_stages:
                # align_corners=False is common for 'bilinear' to avoid checkerboard
                # True might be better for 'nearest' in some cases.
                decoder_blocks_list.append(
                    nn.Upsample(
                        scale_factor=2,
                        mode=upsample_mode,
                        align_corners=(
                            upsample_mode != "bilinear" and upsample_mode != "bicubic"
                        ),
                    )
                )
                # Note: Upsample doesn't change channels. The subsequent ResBlock will handle channel changes.

            decoder_blocks_list.append(
                ResidualBlock(
                    in_channels=in_c,  # in_c from previous stage or upsampled bottleneck
                    out_channels=out_c,
                    use_batch_norm=use_batch_norm,
                    activation_str=activation_str,
                )
            )

        decoder_blocks_list.append(nn.Tanh())
        self.decoder_conv = nn.Sequential(*decoder_blocks_list)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder_conv(x)
        h_flat = rearrange(h, "b c h w -> b (c h w)")
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h_flat = self.decoder_input_fc(z)
        h = rearrange(
            h_flat,
            "b (c h w) -> b c h w",
            c=self._encoder_output_shape[1],
            h=self._encoder_output_shape[2],
            w=self._encoder_output_shape[3],
        )
        return self.decoder_conv(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
