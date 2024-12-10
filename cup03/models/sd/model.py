import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import get_ipython
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from .config import DiffusionModelConfig, VAEModelConfig
from .ddim import DDIMSampler
from .ddpm import DDPMSampler
from .ema import EMAStateDict
from .modules import Diffusion, FrozenOpenCLIPEmbedder, VAEDecoder, VAEEncoder


class VAEModel:
    def __init__(
        self,
        cfg: VAEModelConfig,
    ):
        self.cfg = cfg
        self.encoder = (
            VAEEncoder(
                input_channels=3,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dim,
                scale=cfg.vae_scale,
            )
            .to(cfg.device)
            .train()
        )
        self.decoder = (
            VAEDecoder(
                output_channels=3,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dim,
                scale=cfg.vae_scale,
            )
            .to(cfg.device)
            .train()
        )

        self.optimizer = optim.Adam(
            [
                *self.encoder.parameters(),
                *self.decoder.parameters(),
            ],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        if cfg.mixed_precision:
            self.scaler = GradScaler(cfg.device_type)
        else:
            self.scaler = None

    def generate(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate images from the given images.

        :param images torch.Tensor: The images tensor of shape (batch_size, 3, h, w)
        :return torch.Tensor: The generated images tensor of shape (batch_size, 3, h, w)
        """
        images = images.to(self.cfg.device)
        latents = self.encoder(images)
        pred_images = self.decoder(latents)
        pred_images = torch.clamp(
            pred_images, self.cfg.image_range[0], self.cfg.image_range[1]
        )
        return pred_images

    def compute_loss(
        self,
        images: torch.Tensor,
        pred_images: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ):
        image_loss = F.mse_loss(pred_images, images)

        # KL divergence
        # Clamp the log variance to avoid numerical instability
        log_variance = torch.clamp(log_variance, -30, 20)
        kl_loss = -0.5 * torch.sum(
            1 + log_variance - mean.pow(2) - log_variance.exp(), dim=(1, 2, 3)
        )
        kl_loss = kl_loss.mean()

        loss = image_loss + self.cfg.vae_beta * kl_loss
        return loss

    def train_step(self, images: torch.Tensor):
        """
        Perform a training step.

        :param images torch.Tensor: The images tensor of shape (batch_size, 3, h, w)
        :return Dict[str, torch.Tensor]: The metrics dictionary
        """
        with autocast(
            self.cfg.device_type,
            enabled=self.cfg.mixed_precision,
        ):
            images = images.to(self.cfg.device)
            mean, log_variance = self.encoder.encode(images)
            latents = self.encoder.reparametrize(mean, log_variance)
            pred_images = self.decoder(latents)

            loss = self.compute_loss(images, pred_images, mean, log_variance)

        # Backward pass
        self.optimizer.zero_grad()
        if self.cfg.mixed_precision:
            self.scaler.scale(loss).backward()
            self.clip_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.clip_grad_norm()
            self.optimizer.step()

        return {
            "loss": loss.item(),
        }

    def clip_grad_norm(self, max_norm: float = 1.0):
        """
        Clip the gradient norms of the model.

        :param max_norm float: The maximum norm value
        """
        torch.nn.utils.clip_grad_norm_(
            self.encoder.parameters(), max_norm=max_norm
        )
        torch.nn.utils.clip_grad_norm_(
            self.decoder.parameters(), max_norm=max_norm
        )

    @torch.no_grad()
    def test_step(self, images: torch.Tensor):
        """
        Perform a testing step.

        :param images torch.Tensor: The images tensor of shape (batch_size, 3, h, w)
        :return Dict[str, torch.Tensor]: The metrics dictionary
        """
        images = images.to(self.cfg.device)
        with autocast(
            self.cfg.device_type,
            enabled=self.cfg.mixed_precision,
        ):
            mean, log_variance = self.encoder.encode(images)
            latents = self.encoder.reparametrize(mean, log_variance)
            pred_images = self.decoder(latents)

            loss = self.compute_loss(images, pred_images, mean, log_variance)
        return {
            "loss": loss.item(),
        }

    @torch.no_grad()
    def checkpoint(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Get the state dictionary of the model.

        :return Dict[str, Dict[str, nn.Parameter]]: The state dictionary
        """
        checkpoint = {"optimizer": self.optimizer.state_dict()}
        checkpoint["encoder"] = self.encoder.state_dict()
        checkpoint["decoder"] = self.decoder.state_dict()
        return checkpoint

    @torch.no_grad()
    def load_checkpoint(self, checkpoint: Dict[str, Dict[str, nn.Parameter]]):
        """
        Set the state dictionary of the model.

        :param checkpoint Dict[str, Dict[str, nn.Parameter]]: The state dictionary
        """
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])

    @torch.no_grad()
    def plot_images(
        self,
        images: torch.Tensor,
        num_rows=1,
        num_cols=6,
        save=False,
        output_dir="outputs",
        epoch=None,
    ):
        """
        Plot the images in notebook or save them to the output directory with format `img_epoch{epoch}.png`.

        :param images torch.Tensor: The images tensor of shape (batch_size, 3, h, w)
        :param num_rows int: The number of rows
        :param num_cols int: The number of columns
        :param save bool: Whether to save the images
        :param output_dir str: The output directory
        :param epoch Optional[int]: The epoch number
        """
        ipy = get_ipython()
        if ipy is None and not save:
            return

        with autocast(self.cfg.device_type, enabled=self.cfg.mixed_precision):
            pred_images = self.generate(images)
        # To float32
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred_images = (
            pred_images.detach()
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
            .astype(np.float32)
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 4.0))
        for row in range(num_rows):
            for col in range(num_cols):
                idx = row * 2 * num_cols + col
                if idx >= images.shape[0]:
                    continue
                plt.subplot(2 * num_rows, num_cols, idx + 1)
                plt.imshow(images[idx])
                plt.axis("off")
                plt.subplot(2 * num_rows, num_cols, idx + 1 + num_cols)
                plt.imshow(pred_images[idx])
                plt.axis("off")
        plt.tight_layout()

        # Choose to show or save the images
        if ipy is not None:
            plt.show()
        if save:
            os.makedirs(output_dir, exist_ok=True)
            filename = "img.png" if epoch is None else f"img_epoch{epoch}.png"
            plt.savefig(os.path.join(output_dir, filename))
            print("Saved generated images at", output_dir)
        plt.close()


class DiffusionModel:
    def __init__(
        self,
        cfg: DiffusionModelConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

        self.mean = torch.tensor(
            # [0.4560279846191406, 0.3863838016986847, 0.2983206808567047], # Derived from the training dataset
            [0.5, 0.5, 0.5],  # To [-1, 1]
            device=cfg.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            # [0.2934732437133789, 0.24085882306098938, 0.26899823546409607], # Derived from the training dataset
            [0.5, 0.5, 0.5],  # To [-1, 1]
            device=cfg.device,
        ).view(1, 3, 1, 1)

        self.sampler = self.get_sampler(cfg.max_num_steps)
        self.clip = FrozenOpenCLIPEmbedder(
            precision=cfg.clip_precision,
            device=cfg.device,
        )
        self.diffusion = Diffusion(
            latent_dim=3,
            context_dim=cfg.context_dim,
            hidden_context_dim=cfg.hidden_context_dim,
            time_dim=cfg.time_dim,
            num_heads=cfg.num_heads,
            unet_scale=cfg.unet_scale,
        ).to(cfg.device)

        if cfg.ema_enabled:
            self.diffusion_ema = EMAStateDict(
                state_dict=self.diffusion.state_dict(),
                decay=cfg.ema_decay,
                device=cfg.device,
            )
        else:
            self.diffusion_ema = None

        if cfg.distributed:
            self.diffusion = DDP(
                self.diffusion,
                device_ids=[cfg.device_ids[cfg.local_rank]],
            )

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.diffusion.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        if cfg.mixed_precision:
            self.scaler = GradScaler(cfg.device_type)
        else:
            self.scaler = None

        self.fid = FrechetInceptionDistance(
            feature=64,
            input_img_size=(3, cfg.image_height, cfg.image_width),
        ).to(cfg.device)

    def get_sampler(self, num_steps: int):
        """
        Get the sampler for the given number of steps.

        :param num_steps int: The number of steps
        :return Sampler: The sampler
        """
        if self.cfg.sampler_type == "ddpm":
            return DDPMSampler(
                num_steps=num_steps,
                max_num_steps=self.cfg.max_num_steps,
                device=self.cfg.device,
            )
        elif self.cfg.sampler_type == "ddim":
            return DDIMSampler(
                num_steps=num_steps,
                max_num_steps=self.cfg.max_num_steps,
                device=self.cfg.device,
            )
        else:
            raise ValueError(
                "Unknown sampler value {}".format(self.cfg.sampler_type)
            )

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize the images.

        :param images torch.Tensor: The images tensor of shape (batch_size, 3, h, w)
        :return torch.Tensor: The normalized images tensor of shape (batch_size, 3, h, w)
        """
        return (images - self.mean) / self.std

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the images.

        :param images torch.Tensor: The images tensor of shape (batch_size, 3, h, w)
        :return torch.Tensor: The denormalized images tensor of shape (batch_size, 3, h, w)
        """
        return images * self.std + self.mean

    def compute_fid(self, pred_images: torch.Tensor, images: torch.Tensor):
        """
        Compute the Frechet Inception Distance (FID) between the predicted and real images.

        :param pred_images torch.Tensor: The predicted images tensor of shape (batch_size, 3, h, w)
        :param images torch.Tensor: The real images tensor of shape (batch_size, 3, h, w)
        :return torch.Tensor: The FID value
        """
        self.fid.reset()
        pred_images = self.denormalize(pred_images) * 255
        pred_images = pred_images.to(torch.uint8)
        images = self.denormalize(images) * 255
        images = images.to(torch.uint8)
        self.fid.update(pred_images, real=False)
        self.fid.update(images, real=True)
        fid = self.fid.compute()
        return fid

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        num_steps: int = 50,
        seed: Optional[int] = None,
        progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        Generates an image from the prompts

        :param prompts list[str]: The prompts string
        :param num_steps int: The number of sampling steps
        :param seed Optional[int]: The seed to use
        :return torch.Tensor: The generated image, shape (3, h, w)
        """

        generator = torch.Generator(device=self.cfg.device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # Create the sampler
        sampler = self.get_sampler(num_steps)

        if self.cfg.ema_enabled:
            diffusion = Diffusion(
                latent_dim=3,
                context_dim=self.cfg.context_dim,
                hidden_context_dim=self.cfg.hidden_context_dim,
                time_dim=self.cfg.time_dim,
                num_heads=self.cfg.num_heads,
                unet_scale=self.cfg.unet_scale,
            ).to(self.cfg.device)
            diffusion.load_state_dict(self.diffusion_ema.state_dict())
        else:
            diffusion = self.diffusion
        diffusion.eval()

        batch_size = len(prompts)

        # Encode the prompts
        if self.cfg.offload_clip_to_cpu:
            self.clip.to(self.cfg.device)
        context = self.clip(prompts)  # (batch_size, 77, embed_dim)
        if self.cfg.offload_clip_to_cpu:
            self.clip.to("cpu")

        # Generate noise
        image_shape = (
            batch_size,
            3,
            self.cfg.image_height,
            self.cfg.image_width,
        )
        noise = sampler.sample_noise(
            image_shape, device=self.cfg.device, dtype=torch.float32
        )
        next_noisy_image = noise

        # Diffusion
        step_iters = list(reversed(range(num_steps)))
        if progress_bar:
            step_iters = tqdm(step_iters)
        for step in step_iters:
            steps = torch.tensor([step]).to(self.cfg.device)  # (1,)
            time_embedding = self.get_time_embedding(steps).to(
                self.cfg.device
            )  # (1, time_dim)

            pred_noise = diffusion(
                next_noisy_image, context, time_embedding
            )  # ((1 + cfg_enable) * batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

            pred_image, next_noisy_image = sampler.denoise(
                steps,
                next_noisy_image,
                pred_noise,
            )

        pred_image = self.denormalize(pred_image)
        pred_image = torch.clamp(pred_image, 0, 1)
        return pred_image

    def get_time_embedding(self, steps: torch.Tensor) -> torch.Tensor:
        """
        Get the time embedding for the given steps.

        :param steps torch.Tensor: The steps to get the time embedding for, shape (batch_size,)
        :return torch.Tensor: The time embedding of shape (batch_size, time_dim)
        """
        max_freq = self.cfg.max_embedding_freq
        time_dim = self.cfg.time_dim
        half_num_time = time_dim // 2

        freqs = torch.pow(
            max_freq,
            -torch.arange(
                start=0,
                end=half_num_time,
                dtype=torch.float32,
                device=steps.device,
            )
            / half_num_time,
        )  # Shape: (time_dim/2,)
        time = steps[:, None] * freqs[None]  # Shape: (batch_size, time_dim/2)
        time = torch.cat(
            [torch.cos(time), torch.sin(time)], dim=-1
        )  # Shape: (batch_size, time_dim)
        return time

    def train_step(self, images: torch.Tensor, prompts: List[str]):
        """
        Perform a training step.

        :param images torch.Tensor: The images tensor of shape (batch_size, 3, h, w)
        :param prompts List[str]: The list of prompts, shape (batch_size,)
        :return Dict[str, torch.Tensor]: The metrics dictionary
        """
        assert images.size(0) == len(prompts), "Batch size mismatch"
        batch_size = images.size(0)

        with autocast(
            self.cfg.device_type,
            enabled=self.cfg.mixed_precision,
        ):
            with torch.no_grad():
                # Encode the prompts
                if self.cfg.offload_clip_to_cpu:
                    self.clip.to(self.cfg.device)
                context = self.clip(prompts)
                if self.cfg.offload_clip_to_cpu:
                    self.clip.to(self.cfg.device)

                # Random steps
                steps = torch.randint(
                    0,
                    self.cfg.max_num_steps,
                    (batch_size,),
                    device=self.cfg.device,
                )

                # Normalize the images
                images = images.to(self.cfg.device)
                images = self.normalize(images)

                # Generate noise
                noise = self.sampler.sample_noise(
                    images.shape, device=self.cfg.device, dtype=torch.float32
                )
                noisy_images = self.sampler.add_noise(
                    steps, images, noise
                )  # (batch_size, 3, h, w)
                time_embedding = self.get_time_embedding(steps).to(
                    self.cfg.device
                )  # (batch_size, time_dim)

            self.diffusion.train()
            pred_noises = self.diffusion(
                noisy_images, context, time_embedding
            )  # (batch_size, 3, h, w)
            pred_images, _ = self.sampler.denoise(
                steps, noisy_images, pred_noises
            )  # (batch_size, 3, h, w)

            noise_loss = self.loss_func(pred_noises, noise)
            image_loss = self.loss_func(pred_images, images)

        # Backward pass
        self.optimizer.zero_grad()
        if self.cfg.mixed_precision:
            self.scaler.scale(noise_loss).backward()
            self.clip_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            noise_loss.backward()
            self.clip_grad_norm()
            self.optimizer.step()

        if self.cfg.ema_enabled:
            self.ema_update()

        return {
            "noise_loss": noise_loss.item(),
            "image_loss": image_loss.item(),
        }

    def clip_grad_norm(self):
        """
        Clip the gradient norms of the model.

        :param max_norm float: The maximum norm value
        """
        if self.cfg.grad_clip is None:
            return

        torch.nn.utils.clip_grad_norm_(
            self.diffusion.parameters(), max_norm=self.cfg.grad_clip
        )

    @torch.no_grad()
    def ema_update(self):
        """
        Update the EMA models.
        """
        if not self.cfg.ema_enabled:
            return

        self.diffusion_ema.update(self.diffusion.state_dict())

    @torch.no_grad()
    def test_step(self, images: torch.Tensor, prompts: List[str]):
        """
        Perform a testing step.

        :param images torch.Tensor: The images tensor of shape (batch_size, 3, h, w)
        :param prompts List[str]: The list of prompts, shape (batch_size,)
        :return Dict[str, torch.Tensor]: The metrics dictionary
        """
        assert images.size(0) == len(prompts), "Batch size mismatch"
        batch_size = images.size(0)

        with autocast(
            self.cfg.device_type,
            enabled=self.cfg.mixed_precision,
        ):
            # Encode the prompts
            if self.cfg.offload_clip_to_cpu:
                self.clip.to(self.cfg.device)
            context = self.clip(prompts)
            if self.cfg.offload_clip_to_cpu:
                self.clip.to(self.cfg.device)

            # Random steps
            steps = torch.randint(
                0,
                self.cfg.max_num_steps,
                (batch_size,),
                device=self.cfg.device,
            )

            # Normalize the images
            images = images.to(self.cfg.device)
            images = self.normalize(images)

            # Generate noise
            noise = self.sampler.sample_noise(
                images.shape, device=self.cfg.device, dtype=torch.float32
            )
            noisy_images = self.sampler.add_noise(
                steps, images, noise
            )  # (batch_size, 3, h, w)
            time_embedding = self.get_time_embedding(steps).to(
                self.cfg.device
            )  # (batch_size, time_dim)

            self.diffusion.eval()
            pred_noises = self.diffusion(
                noisy_images, context, time_embedding
            )  # (batch_size, 3, h, w)
            pred_images, _ = self.sampler.denoise(
                steps, noisy_images, pred_noises
            )  # (batch_size, 3, h, w)

            noise_loss = self.loss_func(pred_noises, noise)
            image_loss = self.loss_func(pred_images, images)
            fid = self.compute_fid(pred_images, images)

        return {
            "noise_loss": noise_loss.item(),
            "image_loss": image_loss.item(),
            "fid": fid.item(),
        }

    @torch.no_grad()
    def checkpoint(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Get the state dictionary of the model.

        :return Dict[str, Dict[str, nn.Parameter]]: The state dictionary
        """
        checkpoint = {"optimizer": self.optimizer.state_dict()}
        if self.cfg.distributed:
            diffusion = self.diffusion.module
        else:
            diffusion = self.diffusion
        checkpoint["diffusion"] = diffusion.state_dict()
        if self.cfg.ema_enabled:
            checkpoint["diffusion_ema"] = self.diffusion_ema.state_dict()
        return checkpoint

    @torch.no_grad()
    def load_checkpoint(self, checkpoint: Dict[str, Dict[str, nn.Parameter]]):
        """
        Set the state dictionary of the model.

        :param checkpoint Dict[str, Dict[str, nn.Parameter]]: The state dictionary
        """
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.cfg.distributed:
            self.diffusion.module.load_state_dict(checkpoint["diffusion"])
        else:
            self.diffusion.load_state_dict(checkpoint["diffusion"])
        if self.cfg.ema_enabled:
            self.diffusion_ema.load_state_dict(checkpoint["diffusion_ema"])

    @torch.no_grad()
    def plot_images(
        self,
        prompts: List[str] = [],
        num_steps: int = 50,
        num_rows=3,
        num_cols=6,
        save=False,
        output_dir="outputs",
        epoch=None,
        seed=None,
        progress_bar=False,
    ):
        """
        Plot the images in notebook or save them to the output directory with format `img_epoch{epoch}.png`.

        :param num_rows int: The number of rows
        :param num_cols int: The number of columns
        :param save bool: Whether to save the images
        :param output_dir str: The output directory
        :param epoch Optional[int]: The epoch number
        """
        ipy = get_ipython()
        if ipy is None and not save:
            return

        # Pad the prompts until we have enough
        for _ in range(num_rows * num_cols - len(prompts)):
            prompts.append("")

        with autocast(self.cfg.device_type, enabled=self.cfg.mixed_precision):
            generated_images = self.generate(
                prompts,
                num_steps=num_steps,
                seed=seed,
                progress_bar=progress_bar,
            )
        # To float32
        generated_images = (
            generated_images.detach()
            .cpu()
            .permute(0, 2, 3, 1)
            .numpy()
            .astype(np.float32)
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                idx = row * num_cols + col
                plt.subplot(num_rows, num_cols, idx + 1)
                plt.imshow(generated_images[idx])
                plt.axis("off")
        plt.tight_layout()

        # Choose to show or save the images
        if ipy is not None:
            plt.show()
        if save:
            os.makedirs(output_dir, exist_ok=True)
            filename = "img.png" if epoch is None else f"img_epoch{epoch}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            self.logger.info(f"Saved generated images at {filepath}")
        plt.close()
