from dataclasses import dataclass


@dataclass
class BaseModelConfig:
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Runtime parameters
    mixed_precision: bool = True
    device: str = "cuda"
    device_type: str = "cuda"  # "cuda" or "cpu"

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device_ids: list[int] = None
    distributed: bool = False


@dataclass
class VAEModelConfig(BaseModelConfig):
    # Model architecture
    image_height: int = 512
    image_width: int = 512
    latent_height: int = 64
    latent_width: int = 64
    latent_dim: int = 4
    num_heads: int = 8
    vae_widths: tuple[int, int, int] = (64, 128, 256)

    # Training parameters
    ema_enabled: bool = True
    ema_decay: float = 0.999
    weight_kl: float = 1.0  # strength of the KL divergence term


@dataclass
class DiffusionModelConfig(BaseModelConfig):
    # Model architecture
    image_height: int = 64
    image_width: int = 64
    context_dim: int = 1024
    hidden_context_dim: int = 512
    unet_scale: int = 4
    time_dim: int = 320
    num_heads: int = 8

    # Diffusion parameters
    max_embedding_freq: int = 10000.0
    sampler_type: str = "ddpm"
    max_num_steps: int = 1000

    # Training parameters
    ema_enabled: bool = True
    ema_decay: float = 0.999

    # Runtime parameters
    offload_clip_to_cpu: bool = False
    clip_precision: str = "fp32"
