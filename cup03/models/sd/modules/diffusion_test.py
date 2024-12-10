import unittest

import torch

from .attention import set_attn_backend
from .diffusion import Diffusion


class TestDiffusion(unittest.TestCase):
    def setUp(self):
        """Set up the test."""
        self.latent_dim = 256
        self.context_dim = 768
        self.hidden_context_dim = 512
        self.time_dim = 320
        self.num_heads = 8
        self.unet_scale = 4
        self.backend = "xformers"  # Choose between "xformers" and "pytorch"

        self.batch_size = 2
        self.h = 64
        self.w = 64
        self.seq_len = 10

        self.latent = torch.randn(
            self.batch_size, self.latent_dim, self.h // 8, self.w // 8
        ).cuda()
        self.context = torch.randn(
            self.batch_size, self.seq_len, self.context_dim
        ).cuda()
        self.time = torch.randn(self.batch_size, self.time_dim).cuda()

    def test_output_shape(self):
        """Test if the output has the expected shape."""
        set_attn_backend(self.backend)
        diffusion = Diffusion(
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
            hidden_context_dim=self.hidden_context_dim,
            time_dim=self.time_dim,
            num_heads=self.num_heads,
            unet_scale=self.unet_scale,
        ).cuda()
        output = diffusion(self.latent, self.context, self.time)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.latent_dim, self.h // 8, self.w // 8),
        )


if __name__ == "__main__":
    unittest.main()
