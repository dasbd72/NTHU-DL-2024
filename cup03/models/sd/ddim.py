from typing import Optional, Tuple, Union

import torch


class DDIMSampler:
    def __init__(
        self,
        generator: Optional[torch.Generator] = None,
        num_steps: int = 50,
        max_num_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        device: str = "cuda",
    ):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        self.generator = generator
        self.num_steps = num_steps
        self.max_num_steps = max_num_steps

        self.betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                max_num_steps,
                dtype=torch.float32,
                device=device,
            )
            ** 2
        )  # (max_num_steps,)
        self.alphas = 1.0 - self.betas  # (max_num_steps,)
        self.alphas_cumprod = torch.cumprod(
            self.alphas, dim=0
        )  # (max_num_steps,)
        self.stride = self.max_num_steps // self.num_steps

    @torch.no_grad()
    def sample_noise(
        self,
        shape: Tuple[int, int, int, int],
        device: Union[str, torch.device] = "cuda",
        dtype=torch.float32,
    ) -> torch.FloatTensor:
        """
        :param shape Tuple[int, int, int, int]: The shape of the noise tensor, (batch_size, latent_dim, h, w)
        :param device Union[str, torch.device]: The device to create the noise tensor on
        :param dtype torch.dtype: The data type of the noise tensor
        :return torch.FloatTensor: The noise tensor, shape (batch_size, latent_dim, h, w)
        """
        noise = torch.randn(
            shape,
            generator=self.generator,
            dtype=dtype,
            device=device,
        )
        return noise

    @torch.no_grad()
    def add_noise(
        self,
        steps: torch.IntTensor,
        latents: torch.FloatTensor,
        noise: torch.FloatTensor,
    ):
        """
        Add noise to the latents at the given steps.

        :param steps torch.IntTensor: The steps to add noise at, shape (batch_size,)
        :param latents torch.FloatTensor: The latents tensor of shape (batch_size, latent_dim, h, w)
        :param noise torch.FloatTensor: The noise tensor of shape (batch_size, latent_dim, h, w)
        :return torch.FloatTensor: The noisy latents tensor
        """
        assert torch.all(
            (0 <= steps) & (steps < self.num_steps)
        ), "steps must be in the range [0, num_steps)"

        dtype = latents.dtype
        device = latents.device

        # Scale the steps to the range [0, max_num_steps)
        steps = steps * self.stride

        alphas_cumprod = self.alphas_cumprod.to(device, dtype)
        steps = steps.to(device)

        sqrt_alphas_cumprod = alphas_cumprod[steps].sqrt()
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(
            -1, 1, 1, 1
        )  # (batch_size, 1, 1, 1)

        sqrt_one_minus_alphas_cumprod = (
            1 - alphas_cumprod[steps]
        ).sqrt()  # stddev
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(
            -1, 1, 1, 1
        )  # (batch_size, 1, 1, 1)

        noisy_latents = (
            sqrt_alphas_cumprod * latents
            + sqrt_one_minus_alphas_cumprod * noise
        )
        return noisy_latents

    @torch.no_grad()
    def denoise(
        self,
        steps: torch.IntTensor,
        latents: torch.FloatTensor,
        noise: torch.FloatTensor,
    ):
        """
        Denoise the latents at the given step.

        :param steps torch.IntTensor: The steps to denoise at, shape (batch_size,)
        :param latents torch.FloatTensor: The latents tensor of shape (batch_size, latent_dim, h, w)
        :param noise torch.FloatTensor: The output tensor of shape (batch_size, latent_dim, h, w)
        :return torch.FloatTensor: The denoised latents tensor, shape (batch_size, latent_dim, h, w)
        :return Tuple[torch.FloatTensor, torch.FloatTensor]: The predicted original latent and the predicted noisy latents,
            shape (batch_size, latent_dim, h, w).
        """
        if not torch.all((0 <= steps) & (steps < self.num_steps)):
            raise ValueError(
                f"steps must be in the range [0, {self.num_steps})"
            )

        dtype = latents.dtype
        device = latents.device

        # TODO: variable eta
        eta = 0.0

        # Scale the steps to the range [0, max_num_steps)
        prev_step = (steps - 1) * self.stride  # (batch_size,)
        steps = steps * self.stride  # (batch_size,)

        # 1. compute alphas, betas
        alpha_prod = self.alphas_cumprod[steps]  # (batch_size,)
        prev_alphas_prod = torch.where(
            steps > 0,
            self.alphas_cumprod[prev_step.clamp(min=0)],
            torch.ones_like(alpha_prod, device=device, dtype=dtype),
        )
        alpha_prod = alpha_prod.view(-1, 1, 1, 1)  # (batch_size, 1, 1, 1)
        prev_alphas_prod = prev_alphas_prod.view(
            -1, 1, 1, 1
        )  # (batch_size, 1, 1, 1)
        beta_prod = 1 - alpha_prod  # (batch_size, 1, 1, 1)
        prev_beta_prod = 1 - prev_alphas_prod  # (batch_size, 1, 1, 1)
        alpha = alpha_prod / prev_alphas_prod  # (batch_size, 1, 1, 1)

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - (beta_prod**0.5) * noise) / (
            alpha_prod**0.5
        )  # (batch_size, latent_dim, h, w)

        # Compute variance
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = (
            prev_beta_prod / beta_prod * (1 - alpha_prod / prev_alphas_prod)
        )
        stddev = eta * variance.sqrt()

        # Compute direction pointing to x_t
        pred_sample_direction = (
            1 - prev_alphas_prod - stddev**2
        ) ** 0.5 * noise

        # Compute x_t without "random noise"
        pred_prev_sample = (
            prev_alphas_prod.sqrt() * pred_original_sample
            + pred_sample_direction
        )

        if eta > 0:
            noise = self.sample_noise(
                latents.shape, device=device, dtype=dtype
            )
            noise = torch.where(
                (steps > 0).view(-1, 1, 1, 1),
                noise,
                torch.zeros_like(noise, device=device, dtype=dtype),
            )  # (batch_size, latent_dim, h, w)
            noise = (
                variance.sqrt() * eta * noise
            )  # (batch_size, latent_dim, h, w)
            pred_prev_sample = pred_prev_sample + noise

        return pred_original_sample, pred_prev_sample
