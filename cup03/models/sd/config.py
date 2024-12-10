from dataclasses import dataclass


@dataclass
class VAEModelConfig:
    # Model architecture
    image_height: int = 512
    image_width: int = 512
    image_range: tuple[float, float] = (0, 1)
    latent_height: int = 64
    latent_width: int = 64
    latent_range: tuple[float, float] = (-1, 1)

    latent_dim: int = 4
    num_heads: int = 8

    # Training parameters
    vae_scale: int = 4
    vae_beta: float = 1.0  # strength of the KL divergence term
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Runtime parameters
    mixed_precision: bool = False
    device: str = "cuda"
    device_type: str = "cuda"

    rank: int = 0
    world_size: int = 1
    device_ids: list[int] = None
    distributed: bool = False


@dataclass
class DiffusionModelConfig:
    # Model architecture
    image_height: int = 64
    image_width: int = 64

    context_dim: int = 1024
    hidden_context_dim: int = 512
    unet_scale: int = 4
    time_dim: int = 320
    num_heads: int = 8

    max_embedding_freq: int = 10000.0

    # Diffusion parameters
    sampler_type: str = "ddpm"
    max_num_steps: int = 1000

    # Training parameters
    ema_enabled: bool = True
    ema_decay: float = 0.999
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Runtime parameters
    offload_clip_to_cpu: bool = False
    clip_precision: str = "fp32"
    mixed_precision: bool = False
    device: str = "cuda"
    device_type: str = "cuda"  # "cuda" or "cpu"

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device_ids: list[int] = None
    distributed: bool = False
