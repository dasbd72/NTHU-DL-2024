import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention, SelfAttention
from .initialize import UniformInitConv2d, UniformInitLinear


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(UNetResidualBlock, self).__init__()

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = UniformInitConv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )
        self.gn_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = UniformInitConv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.fc_time = UniformInitLinear(time_dim * 4, out_channels)
        self.gn_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = UniformInitConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, feature: torch.Tensor, time: torch.Tensor):
        """
        :param feature torch.Tensor: (batch_size, in_channels, h, w)
        :param time torch.Tensor: (1 or batch_size, time_dim)
        :return torch.Tensor: (batch_size, out_channels, h, w)
        """
        residual = self.residual_layer(
            feature
        )  # (batch_size, out_channels, h, w)
        feature = self.gn_feature(feature)  # (batch_size, in_channels, h, w)
        feature = F.silu(feature)
        feature = self.conv_feature(
            feature
        )  # (batch_size, out_channels, h, w)

        time = F.silu(time)  # (1 or batch_size, time_dim)
        time = self.fc_time(time)  # (1 or batch_size, out_channels)
        time = time.unsqueeze(-1).unsqueeze(
            -1
        )  # (1 or batch_size, out_channels, 1, 1)

        merged = feature + time  # (batch_size, out_channels, h, w)
        merged = self.gn_merged(merged)  # (batch_size, out_channels, h, w)
        merged = F.silu(merged)  # (batch_size, out_channels, h, w)
        merged = self.conv_merged(merged)  # (batch_size, out_channels, h, w)
        out = merged + residual  # (batch_size, out_channels, h, w)
        return out


class UNetAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        context_dim,
        num_heads=8,
    ):
        super(UNetAttentionBlock, self).__init__()

        if channels % num_heads != 0:
            raise ValueError(
                f"channels should be divisible by num_heads. Got channels={channels} and num_heads={num_heads}"
            )

        self.head_dim = channels // num_heads

        self.gn = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_in = UniformInitConv2d(
            channels, channels, kernel_size=1, padding=0
        )

        self.ln1 = nn.LayerNorm(channels)
        self.attn1 = SelfAttention(
            embed_dim=channels,
            num_heads=num_heads,
            in_proj_bias=False,
        )
        self.ln2 = nn.LayerNorm(channels)
        self.attn2 = CrossAttention(
            context_dim=context_dim,
            latent_dim=channels,
            embed_dim=channels,
            num_heads=num_heads,
            in_proj_bias=False,
        )
        self.ln3 = nn.LayerNorm(channels)
        self.fc1 = UniformInitLinear(channels, 4 * 2 * channels)
        self.fc2 = UniformInitLinear(4 * channels, channels)

        self.conv_out = UniformInitConv2d(
            channels, channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, channels, h, w)
        :param context torch.Tensor: (batch_size, seq_len, context_dim)
        """
        batch_size, channels, h, w = x.shape

        residual_long = x
        x = self.gn(x)  # (batch_size, channels, h, w)
        x = F.silu(x)  # (batch_size, channels, h, w)
        x = self.conv_in(x)  # (batch_size, channels, h, w)
        x = x.view(batch_size, channels, h * w)  # (batch_size, channels, h*w)
        x = x.permute(0, 2, 1)  # (batch_size, h*w, channels)

        # Normalize + self-attention
        residual_short = x
        x = self.ln1(x)  # (batch_size, h*w, channels)
        x = self.attn1(x)  # (batch_size, h*w, channels)
        x = x + residual_short  # (batch_size, h*w, channels)

        # Normalize + cross-attention
        residual_short = x
        x = self.ln2(x)  # (batch_size, h*w, channels)
        x = self.attn2(x, context)  # (batch_size, h*w, channels)
        x = x + residual_short  # (batch_size, h*w, channels)

        # Normalize + FFN with GeGLU and skip connection
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        residual_short = x
        x = self.ln3(x)  # (batch_size, h*w, channels)
        x = self.fc1(x)  # (batch_size, h*w, 4 * 2 * channels)
        x, gate = x.chunk(
            2, dim=-1
        )  # (batch_size, h*w, 4 * channels), (batch_size, h*w, 4 * channels)
        x = x * F.gelu(gate)  # (batch_size, h*w, 4 * channels)
        x = self.fc2(x)  # (batch_size, h*w, channels)
        x = x + residual_short  # (batch_size, h*w, channels)
        x = x.permute(0, 2, 1)  # (batch_size, channels, h*w)
        x = x.view(batch_size, channels, h, w)  # (batch_size, channels, h, w)
        x = self.conv_out(x)  # (batch_size, channels, h, w)

        # Skip connection
        x = x + residual_long  # (batch_size, channels, h, w)

        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.conv = UniformInitConv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, in_channels, h, w)
        :return torch.Tensor: (batch_size, out_channels, 2 * h, 2 * w)
        """
        x = F.interpolate(
            x, scale_factor=2, mode="nearest"
        )  # (batch_size, in_channels, 2 * h, 2 * w)
        x = self.conv(x)  # (batch_size, out_channels, 2 * h, 2 * w)
        return x


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        """
        :param x torch.Tensor: (batch_size, in_channels, h, w)
        :param context torch.Tensor: (batch_size, seq_len, context_dim)
        :param time torch.Tensor: (1 or batch_size, time_dim)
        """
        for module in self:
            if isinstance(module, UNetAttentionBlock):
                x = module(x, context)
            elif isinstance(module, UNetResidualBlock):
                x = module(x, time)
            else:
                x = module(x)
        return x


class UNet(nn.Module):
    def __init__(
        self, latent_dim, time_dim, context_dim, num_heads=8, scale=4
    ):
        """
        :param latent_dim int: The latent dimension
        :param time_dim int: The time dimension
        :param context_dim int: The context dimension
        :param num_heads int: The number of heads in the attention mechanism
        :param scale int: An extra parameter for the UNet to reduce the model size, ranges from 1 to 10 (or more)
        """
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                SwitchSequential(
                    UniformInitConv2d(
                        latent_dim, scale * 32, kernel_size=3, padding=1
                    )
                ),  # h, w
                SwitchSequential(
                    UNetResidualBlock(scale * 32, scale * 32, time_dim),
                    UNetAttentionBlock(
                        scale * 32, context_dim, num_heads=num_heads
                    ),
                ),  # h, w
                SwitchSequential(
                    UNetResidualBlock(scale * 32, scale * 32, time_dim),
                    UNetAttentionBlock(
                        scale * 32, context_dim, num_heads=num_heads
                    ),
                ),  # h, w
                SwitchSequential(
                    UniformInitConv2d(
                        scale * 32,
                        scale * 32,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),  # h/2, w/2
                SwitchSequential(
                    UNetResidualBlock(scale * 32, scale * 64, time_dim),
                    UNetAttentionBlock(
                        scale * 64, context_dim, num_heads=num_heads
                    ),
                ),  # h/2, w/2
                SwitchSequential(
                    UNetResidualBlock(scale * 64, scale * 64, time_dim),
                    UNetAttentionBlock(
                        scale * 64, context_dim, num_heads=num_heads
                    ),
                ),  # h/2, w/2
                SwitchSequential(
                    UniformInitConv2d(
                        scale * 64,
                        scale * 64,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),  # h/4, w/4
                SwitchSequential(
                    UNetResidualBlock(scale * 64, scale * 128, time_dim),
                    UNetAttentionBlock(
                        scale * 128, context_dim, num_heads=num_heads
                    ),
                ),  # h/4, w/4
                SwitchSequential(
                    UNetResidualBlock(scale * 128, scale * 128, time_dim),
                    UNetAttentionBlock(
                        scale * 128, context_dim, num_heads=num_heads
                    ),
                ),  # h/4, w/4
                SwitchSequential(
                    UniformInitConv2d(
                        scale * 128,
                        scale * 128,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),  # h/8, w/8
                SwitchSequential(
                    UNetResidualBlock(scale * 128, scale * 128, time_dim),
                ),  # h/8, w/8
                SwitchSequential(
                    UNetResidualBlock(scale * 128, scale * 128, time_dim),
                ),  # h/8, w/8
            ]
        )

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(scale * 128, scale * 128, time_dim),  # h/8, w/8
            UNetAttentionBlock(
                scale * 128, context_dim, num_heads=num_heads
            ),  # h/8, w/8
            UNetResidualBlock(scale * 128, scale * 128, time_dim),  # h/8, w/8
        )

        self.decoders = nn.ModuleList(
            [
                SwitchSequential(
                    UNetResidualBlock(scale * 256, scale * 128, time_dim),
                ),  # h/8, w/8
                SwitchSequential(
                    UNetResidualBlock(scale * 256, scale * 128, time_dim),
                ),  # h/8, w/8
                SwitchSequential(
                    UNetResidualBlock(scale * 256, scale * 128, time_dim),
                    Upsample(scale * 128, scale * 128),
                ),  # h/4, w/4
                SwitchSequential(
                    UNetResidualBlock(scale * 256, scale * 128, time_dim),
                    UNetAttentionBlock(
                        scale * 128, context_dim, num_heads=num_heads
                    ),
                ),  # h/4, w/4
                SwitchSequential(
                    UNetResidualBlock(scale * 256, scale * 128, time_dim),
                    UNetAttentionBlock(
                        scale * 128, context_dim, num_heads=num_heads
                    ),
                ),  # h/4, w/4
                SwitchSequential(
                    UNetResidualBlock(scale * 192, scale * 128, time_dim),
                    UNetAttentionBlock(
                        scale * 128, context_dim, num_heads=num_heads
                    ),
                    Upsample(scale * 128, scale * 128),
                ),  # h/2, w/2
                SwitchSequential(
                    UNetResidualBlock(scale * 192, scale * 64, time_dim),
                    UNetAttentionBlock(
                        scale * 64, context_dim, num_heads=num_heads
                    ),
                ),  # h/2, w/2
                SwitchSequential(
                    UNetResidualBlock(scale * 128, scale * 64, time_dim),
                    UNetAttentionBlock(
                        scale * 64, context_dim, num_heads=num_heads
                    ),
                ),  # h/2, w/2
                SwitchSequential(
                    UNetResidualBlock(scale * 96, scale * 64, time_dim),
                    UNetAttentionBlock(
                        scale * 64, context_dim, num_heads=num_heads
                    ),
                    Upsample(scale * 64, scale * 64),
                ),  # h, w
                SwitchSequential(
                    UNetResidualBlock(scale * 96, scale * 32, time_dim),
                    UNetAttentionBlock(
                        scale * 32, context_dim, num_heads=num_heads
                    ),
                ),  # h, w
                SwitchSequential(
                    UNetResidualBlock(scale * 64, scale * 32, time_dim),
                    UNetAttentionBlock(
                        scale * 32, context_dim, num_heads=num_heads
                    ),
                ),  # h, w
                SwitchSequential(
                    UNetResidualBlock(scale * 64, scale * 32, time_dim),
                    UNetAttentionBlock(
                        scale * 32, context_dim, num_heads=num_heads
                    ),
                ),  # h, w
            ]
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ):
        """
        :param x torch.Tensor: (batch_size, latent_dim, h, w)
        :param context torch.Tensor: (batch_size, seq_len, context_dim)
        :param time torch.Tensor: (1 or batch_size, time_dim)
        :return torch.Tensor: (batch_size, scale*32, h, w)
        """
        skips = []
        for encoder in self.encoders:
            x = encoder(x, context, time)
            skips.append(x)

        x = self.bottleneck(x, context, time)

        for decoder in self.decoders:
            x = torch.cat([x, skips.pop()], dim=1)
            x = decoder(x, context, time)

        return x


class UNetOutputBlock(nn.Module):
    def __init__(self, out_channels, scale=4):
        super(UNetOutputBlock, self).__init__()

        self.gn = nn.GroupNorm(32, scale * 32)
        self.conv = UniformInitConv2d(
            scale * 32, out_channels, kernel_size=3, padding=1, init_scale=0
        )

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, scale*32, h, w)
        :return torch.Tensor: (batch_size, out_channels, h, w)
        """

        x = self.gn(x)  # (batch_size, scale*32, h, w)
        x = F.silu(x)  # (batch_size, scale*32, h, w)
        x = self.conv(x)  # (batch_size, out_channels, h, w)
        return x
