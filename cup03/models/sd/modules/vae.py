import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VAEResidualBlock, self).__init__()

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, in_channels, h, w)
        :return torch.Tensor: (batch_size, out_channels, h, w)
        """
        residual = self.residual_layer(x)  # (batch_size, out_channels, h, w)
        x = self.norm1(x)  # (batch_size, in_channels, h, w)
        x = F.silu(x)
        x = self.conv1(x)  # (batch_size, out_channels, h, w)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)  # (batch_size, out_channels, h, w)
        return x + residual


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super(VAEAttentionBlock, self).__init__()

        self.norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(embed_dim=channels, num_heads=num_heads)

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, channels, h, w)
        :return torch.Tensor: (batch_size, channels, h, w)
        """
        batch_size, channels, h, w = x.shape

        residual = x  # (batch_size, channels, h, w)

        x = self.norm(x)  # (batch_size, channels, h, w)
        x = x.view(batch_size, channels, h * w)  # (batch_size, channels, h*w)
        x = x.permute(0, 2, 1)  # (batch_size, h*w, channels)
        x = self.attention(x)  # (batch_size, h*w, channels)
        x = x.permute(0, 2, 1)  # (batch_size, channels, h*w)
        x = x.view(batch_size, channels, h, w)  # (batch_size, channels, h, w)

        x = x + residual  # (batch_size, channels, h, w)
        return x


class VAEEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        num_heads=8,
        latent_dim=4,
        scale=4,
    ):
        super(VAEEncoder, self).__init__()

        self.latent_encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, scale * 16, kernel_size=3, padding=1
            ),  # h, w
            VAEResidualBlock(scale * 16, scale * 16),
            VAEResidualBlock(scale * 16, scale * 16),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # h/2 + 1, w/2 + 1
            nn.Conv2d(
                scale * 16, scale * 16, kernel_size=3, stride=2, padding=0
            ),  # h/2, w/2
            VAEResidualBlock(scale * 16, scale * 32),
            VAEResidualBlock(scale * 32, scale * 32),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # h/2 + 1, w/2 + 1
            nn.Conv2d(
                scale * 32, scale * 32, kernel_size=3, stride=2, padding=0
            ),  # h/4, w/4
            VAEResidualBlock(scale * 32, scale * 64),
            VAEResidualBlock(scale * 64, scale * 64),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # h/2 + 1, w/2 + 1
            nn.Conv2d(
                scale * 64, scale * 64, kernel_size=3, stride=2, padding=0
            ),  # h/8, w/8
            VAEResidualBlock(scale * 64, scale * 64),
            VAEResidualBlock(scale * 64, scale * 64),
            VAEResidualBlock(scale * 64, scale * 64),
            VAEAttentionBlock(scale * 64, num_heads=num_heads),
            VAEResidualBlock(scale * 64, scale * 64),
            nn.GroupNorm(32, scale * 64),
            nn.SiLU(),
            nn.Conv2d(scale * 64, latent_dim * 2, kernel_size=3, padding=1),
            nn.Conv2d(
                latent_dim * 2, latent_dim * 2, kernel_size=1, padding=0
            ),  # latent space
        )  # (batch_size, latent_dim*2, h/8, w/8)

    def encode(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, input_channels, h, w)
        :param noise torch.Tensor: (batch_size, latent_dim, h/8, w/8)
        :return torch.Tensor: (batch_size, latent_dim, h/8, w/8)
        """
        x = self.latent_encoder(x)  # (batch_size, latent_dim*2, h/8, w/8)
        mean, log_variance = torch.chunk(
            x, 2, dim=1
        )  # (batch_size, latent_dim, h/8, w/8), (batch_size, latent_dim, h/8, w/8)
        return mean, log_variance

    def reparametrize(self, mean, log_variance):
        """
        :param mean torch.Tensor: (batch_size, latent_dim, h/8, w/8)
        :param log_variance torch.Tensor: (batch_size, latent_dim, h/8, w/8)
        :return torch.Tensor: (batch_size, latent_dim, h/8, w/8)
        """
        noise = torch.randn_like(
            mean, device=mean.device
        )  # (batch_size, latent_dim, h/8, w/8)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8.
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = torch.exp(
            log_variance
        )  # (batch_size, latent_dim, h/8, w/8)
        stddev = torch.sqrt(variance)  # (batch_size, latent_dim, h/8, w/8)
        # Transform distribution from N(0, 1) to N(mean, stddev)
        x = mean + stddev * noise  # (batch_size, latent_dim, h/8, w/8)

        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        return x

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, input_channels, h, w)
        :return torch.Tensor: (batch_size, latent_dim, h/8, w/8)
        """
        mean, log_variance = self.encode(x)
        latent = self.reparametrize(mean, log_variance)
        return latent


class VAEDecoder(nn.Module):
    def __init__(
        self,
        output_channels,
        num_heads=8,
        latent_dim=4,
        scale=4,
    ):
        super(VAEDecoder, self).__init__()

        self.latent_decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding=0),
            nn.Conv2d(latent_dim, scale * 64, kernel_size=3, padding=1),
            VAEResidualBlock(scale * 64, scale * 64),
            VAEAttentionBlock(scale * 64, num_heads=num_heads),
            VAEResidualBlock(scale * 64, scale * 64),
            VAEResidualBlock(scale * 64, scale * 64),
            VAEResidualBlock(scale * 64, scale * 64),
            VAEResidualBlock(scale * 64, scale * 64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # h/4, w/4
            VAEResidualBlock(scale * 64, scale * 64),
            VAEResidualBlock(scale * 64, scale * 64),
            VAEResidualBlock(scale * 64, scale * 64),
            nn.Upsample(scale_factor=2, mode="nearest"),  # h/2, w/2
            nn.Conv2d(scale * 64, scale * 64, kernel_size=3, padding=1),
            VAEResidualBlock(scale * 64, scale * 32),
            VAEResidualBlock(scale * 32, scale * 32),
            VAEResidualBlock(scale * 32, scale * 32),
            nn.Upsample(scale_factor=2, mode="nearest"),  # h, w
            nn.Conv2d(scale * 32, scale * 32, kernel_size=3, padding=1),
            VAEResidualBlock(scale * 32, scale * 16),
            VAEResidualBlock(scale * 16, scale * 16),
            VAEResidualBlock(scale * 16, scale * 16),
            nn.GroupNorm(32, scale * 16),
            nn.SiLU(),
            nn.Conv2d(scale * 16, output_channels, kernel_size=3, padding=1),
        )  # (batch_size, output_channels, h, w)

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, latent_dim, h/8, w/8)
        :return torch.Tensor: (batch_size, output_channels, h, w)
        """

        # Remove the scaling added by the Encoder.
        x = x / 0.18215
        x = self.latent_decoder(x)  # (batch_size, output_channels, h, w)
        return x
