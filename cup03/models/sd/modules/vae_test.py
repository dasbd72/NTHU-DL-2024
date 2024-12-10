import unittest

import torch

from .attention import set_attn_backend
from .vae import VAEDecoder, VAEEncoder


class TestVAEEncoder(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.seed = 42
        self.input_channels = 3  # Example input channels (e.g., RGB image)
        self.latent_dim = 4
        self.batch_size = 16
        self.scale = 4
        self.height = 64  # Example image height
        self.width = 64  # Example image width
        self.noise_shape = (
            self.batch_size,
            self.latent_dim,
            self.height // 8,
            self.width // 8,
        )  # noise shape for VAE

        # Create a random tensor for input (e.g., RGB image)
        self.x = torch.randn(
            self.batch_size, self.input_channels, self.height, self.width
        ).cuda()

    def test_output_shape(self):
        """Test if the output has the expected shape."""
        vae_encoder = VAEEncoder(
            input_channels=self.input_channels,
            latent_dim=self.latent_dim,
            scale=self.scale,
        ).cuda()

        output = vae_encoder(self.x)

        # Expected output shape: (batch_size, self.latent_dim, h/8, w/8)
        self.assertEqual(
            output.shape,
            (
                self.batch_size,
                self.latent_dim,
                self.height // 8,
                self.width // 8,
            ),
        )

    def test_latent_encoder_shape(self):
        """Test if latent_encoder produces the expected shape."""
        vae_encoder = VAEEncoder(
            input_channels=self.input_channels,
            latent_dim=self.latent_dim,
            scale=self.scale,
        ).cuda()

        # Check the output of the latent_encoder before the noise transformation
        output = vae_encoder.latent_encoder(self.x)

        # Expected output shape after latent_encoder: (batch_size, 8, h/8, w/8)
        self.assertEqual(
            output.shape,
            (self.batch_size, 8, self.height // 8, self.width // 8),
        )

    def test_forward_with_different_backends(self):
        """Test forward pass with different backends (if available)."""
        set_attn_backend("pytorch")
        vae_encoder = VAEEncoder(
            input_channels=self.input_channels,
            latent_dim=self.latent_dim,
            scale=self.scale,
        ).cuda()

        # Test with PyTorch backend
        torch.manual_seed(self.seed)
        output_pytorch = vae_encoder(self.x)

        # You could set different backend here if needed (assuming 'xformers' backend is an option)
        set_attn_backend("xformers")
        torch.manual_seed(self.seed)
        output_xformers = vae_encoder(self.x)

        # Assert that the output is the same with both backends (if applicable)
        self.assertTrue(
            torch.allclose(output_pytorch, output_xformers, atol=1e-5)
        )

        # For now, since we only have the PyTorch backend, we only compare within it:
        self.assertEqual(
            output_pytorch.shape,
            (
                self.batch_size,
                self.latent_dim,
                self.height // 8,
                self.width // 8,
            ),
        )


class TestVAEDecoder(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.seed = 42
        self.output_channels = 3  # Example output channels (e.g., RGB image)
        self.latent_dim = 4
        self.batch_size = 16
        self.scale = 4
        self.height = 64  # Example image height
        self.width = 64  # Example image width

        # Create a random tensor for input (latent space)
        self.x = torch.randn(
            self.batch_size,
            self.latent_dim,
            self.height // 8,
            self.width // 8,
        ).cuda()

    def test_output_shape(self):
        """Test if the output has the expected shape."""
        set_attn_backend("pytorch")
        vae_decoder = VAEDecoder(
            output_channels=self.output_channels,
            latent_dim=self.latent_dim,
            scale=self.scale,
        ).cuda()

        output = vae_decoder(self.x)

        # Expected output shape: (batch_size, output_channels, h, w)
        self.assertEqual(
            output.shape,
            (
                self.batch_size,
                self.output_channels,
                self.height,
                self.width,
            ),
        )

        set_attn_backend("xformers")
        vae_decoder = VAEDecoder(
            output_channels=self.output_channels,
            latent_dim=self.latent_dim,
            scale=self.scale,
        ).cuda()

        output = vae_decoder(self.x)

        # Expected output shape: (batch_size, output_channels, h, w)
        self.assertEqual(
            output.shape,
            (
                self.batch_size,
                self.output_channels,
                self.height,
                self.width,
            ),
        )

    def test_latent_decoder_shape(self):
        """Test if latent_decoder produces the expected shape."""
        set_attn_backend("pytorch")
        vae_decoder = VAEDecoder(
            output_channels=self.output_channels,
            latent_dim=self.latent_dim,
            scale=self.scale,
        ).cuda()

        output = vae_decoder.latent_decoder(self.x)

        # Expected output shape after latent_decoder: (batch_size, output_channels, h, w)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.output_channels, self.height, self.width),
        )

    def test_forward_with_different_backends(self):
        """Test forward pass with different backends (if available)."""
        set_attn_backend("pytorch")
        vae_decoder = VAEDecoder(
            output_channels=self.output_channels,
            latent_dim=self.latent_dim,
            scale=self.scale,
        ).cuda()

        # Test with PyTorch backend
        output_pytorch = vae_decoder(self.x)

        # You could set different backend here if needed (assuming 'xformers' backend is an option)
        set_attn_backend("xformers")
        output_xformers = vae_decoder(self.x)

        # Assert that the output is the same with both backends (if applicable)
        self.assertTrue(
            torch.allclose(output_pytorch, output_xformers, atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
