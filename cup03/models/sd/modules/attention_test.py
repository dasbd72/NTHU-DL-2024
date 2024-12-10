import unittest

import torch

from .attention import CrossAttention, SelfAttention, set_attn_backend


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.seed = 42
        self.embed_dim = 64
        self.num_heads = 8
        self.batch_size = 16
        self.seq_len = 20  # Length of the input sequence
        self.backend = "xformers"  # Choose between "xformers" and "pytorch"

        # Create a random tensor for the input (self-attention uses the same input for q, k, and v)
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.embed_dim
        ).cuda()

    def test_output_shape(self):
        """Test if the output has the expected shape."""
        set_attn_backend(self.backend)
        attn_layer = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,
        ).cuda()

        output = attn_layer(self.x)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.embed_dim),
        )

    def test_invalid_embed_dim(self):
        """Test if the module raises an error when embed_dim is not divisible by num_heads."""
        set_attn_backend(self.backend)
        with self.assertRaises(ValueError):
            SelfAttention(
                embed_dim=65,  # Invalid: 65 is not divisible by 8
                num_heads=self.num_heads,
                dropout=0.0,
            )

    def test_forward_without_xformers(self):
        """Test forward pass without xFormers (falling back to PyTorch's attention)."""
        set_attn_backend("pytorch")
        attn_layer = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,
        ).cuda()

        output = attn_layer(self.x)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.embed_dim),
        )

    def test_forward_with_xformers(self):
        """Test forward pass with xFormers (if available)."""
        set_attn_backend("pytorch")
        attn_layer = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,
        ).cuda()

        pytorch_output = attn_layer(self.x)

        set_attn_backend("xformers")  # Switch to xformers backend
        output = attn_layer(self.x)

        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.embed_dim),
        )

        # Check if the output is the same
        self.assertTrue(torch.allclose(output, pytorch_output, atol=1e-5))

    def test_causal_forward_with_xformers(self):
        """Test forward pass with xFormers (if available)."""
        set_attn_backend("pytorch")
        attn_layer = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,
        ).cuda()

        pytorch_output = attn_layer(self.x, causal_mask=True)

        set_attn_backend("xformers")  # Switch to xformers backend
        output = attn_layer(self.x, causal_mask=True)

        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.embed_dim),
        )

        # Check if the output is the same
        self.assertTrue(torch.allclose(output, pytorch_output, atol=1e-5))

    def test_backward(self):
        """Test if the module can perform a backward pass."""
        set_attn_backend("pytorch")  # Switch to PyTorch backend
        attn_layer = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,
        ).cuda()

        output = attn_layer(self.x)
        output.sum().backward()

        set_attn_backend("xformers")  # Switch to xformers backend
        attn_layer = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=0.0,
        ).cuda()

        output = attn_layer(self.x)
        output.sum().backward()


class TestCrossAttention(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.seed = 42
        self.context_dim = 64
        self.latent_dim = 128
        self.embed_dim = 64
        self.num_heads = 8
        self.batch_size = 16
        self.latent_seq_len = 10
        self.context_seq_len = 20
        self.backend = "xformers"

        # Create random tensors for the latent and context
        self.latent = torch.randn(
            self.batch_size, self.latent_seq_len, self.latent_dim
        ).cuda()
        self.context = torch.randn(
            self.batch_size, self.context_seq_len, self.context_dim
        ).cuda()

    def test_output_shape(self):
        """Test if the output has the expected shape."""
        set_attn_backend(self.backend)
        attn_layer = CrossAttention(
            context_dim=self.context_dim,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        ).cuda()
        output = attn_layer(self.latent, self.context)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.latent_seq_len, self.latent_dim),
        )

    def test_invalid_embed_dim(self):
        """Test if the module raises an error when embed_dim is not divisible by num_heads."""
        set_attn_backend(self.backend)
        with self.assertRaises(ValueError):
            CrossAttention(
                context_dim=self.context_dim,
                latent_dim=self.latent_dim,
                embed_dim=65,  # Invalid: 65 is not divisible by 8
                num_heads=self.num_heads,
            )

    def test_forward_without_xformers(self):
        """Test forward pass without xFormers (falling back to PyTorch's attention)."""
        set_attn_backend("pytorch")
        attn_layer = CrossAttention(
            context_dim=self.context_dim,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        ).cuda()
        output = attn_layer(self.latent, self.context)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.latent_seq_len, self.latent_dim),
        )

    def test_forward_with_xformers(self):
        """Test forward pass with xFormers (if available)."""
        set_attn_backend("pytorch")
        attn_layer = CrossAttention(
            context_dim=self.context_dim,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        ).cuda()
        pytorch_output = attn_layer(self.latent, self.context)

        set_attn_backend("xformers")  # Switch to xformers backend
        output = attn_layer(self.latent, self.context)
        self.assertEqual(
            output.shape,
            (self.batch_size, self.latent_seq_len, self.latent_dim),
        )

        # Check if the output is the same
        self.assertTrue(torch.allclose(output, pytorch_output, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
