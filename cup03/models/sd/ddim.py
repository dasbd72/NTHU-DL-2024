from typing import Optional, Tuple, Union

import torch


class DDIMSampler:
    def __init__(
        self,
        generator: Optional[torch.Generator] = None,
        num_steps: int = 50,
        max_num_steps: int = 1000,
        min_signal_rate: float = 0.02,
        max_signal_rate: float = 0.95,
        device: str = "cuda",
    ):
        self.generator = generator
        self.num_steps = num_steps
        self.max_num_steps = max_num_steps
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate

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
    def diffusion_schedule(
        self, steps: torch.IntTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Calculate the noise and signal rates at the given steps.

        :param steps torch.IntTensor: The steps to calculate the noise and signal rates at, shape (batch_size, 1, 1, 1)
        :return tuple[torch.FloatTensor, torch.FloatTensor]: The noise rates and signal rates at the given steps
        """
        # Scale the steps to the range [0, 1]
        steps = steps.float() / self.num_steps
        steps = torch.clamp(steps, 0.0, 1.0)
        max_angle = torch.acos(torch.tensor(self.min_signal_rate)).to(
            steps.device
        )
        min_angle = torch.acos(torch.tensor(self.max_signal_rate)).to(
            steps.device
        )

        # calculate the angles
        angles = steps * (max_angle - min_angle) + min_angle
        # calculate the noise and signal rates
        noise_rates = torch.sin(angles)
        signal_rates = torch.cos(angles)

        return noise_rates, signal_rates

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

        steps = steps.view(-1, 1, 1, 1)  # (batch_size, 1, 1, 1)
        noise_rates, signal_rates = self.diffusion_schedule(steps)
        noisy_latents = signal_rates * latents + noise_rates * noise

        return noisy_latents

    @torch.no_grad()
    def denoise(
        self,
        steps: torch.IntTensor,
        latents: torch.FloatTensor,
        noise: torch.FloatTensor,
    ):
        """
        Denoise the latents at the given steps.

        :param steps torch.IntTensor: The steps to denoise at, shape (batch_size,)
        :param latents torch.FloatTensor: The latents tensor of shape (batch_size, latent_dim, h, w)
        :param noise torch.FloatTensor: The predicted noise, (batch_size, latent_dim, h, w)
        :return Tuple[torch.FloatTensor, torch.FloatTensor]: The predicted original latent and the predicted noisy latents,
            shape (batch_size, latent_dim, h, w).
        """
        assert torch.all(
            (0 <= steps) & (steps < self.num_steps)
        ), "steps must be in the range [0, num_steps)"

        steps = steps.view(-1, 1, 1, 1)

        # Derive the original latent from the predicted noise
        noise_rates, signal_rates = self.diffusion_schedule(steps)
        pred_original_latent = (latents - noise * noise_rates) / signal_rates

        # Remix the noise for the next step
        prev_steps = (steps - 1).clamp(0, self.num_steps - 1)
        pred_noisy_latents = self.add_noise(
            prev_steps, pred_original_latent, noise
        )
        # No more noise after the last step
        pred_noisy_latents = torch.where(
            steps == 0, pred_original_latent, pred_noisy_latents
        )
        return pred_original_latent, pred_noisy_latents
