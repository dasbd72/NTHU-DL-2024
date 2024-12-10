import unittest

import torch

from .clip import FrozenOpenCLIPEmbedder


class TestFrozenOpenCLIPEmbedder(unittest.TestCase):
    def setUp(self):
        """Set up the test case"""
        # Test parameters
        self.text_input = "A sample sentence for testing OpenCLIP."
        self.text_inputs = [
            "A sample sentence for testing OpenCLIP.",
            "Another sample sentence for testing OpenCLIP.",
        ]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = FrozenOpenCLIPEmbedder(
            arch="ViT-H-14-378-quickgelu",  # Modify if necessary
            version="dfn5b",  # Modify if necessary
            max_length=77,
            freeze=True,
            precision="fp32",  # Change to "pure_fp16" for FP16 precision
            layer="last",  # Can test with "penultimate" layer as well
            device=self.device,
        )

    def test_forward_output_shape(self):
        """Test the output shape of the forward pass"""
        # Make sure to run on the correct device
        self.embedder = self.embedder.to(self.device)

        # Run the forward pass
        output = self.embedder(self.text_input)

        # Assert the output shape (batch_size, seq_len, embed_dim)
        # For the `ViT-H-14-378-quickgelu` architecture, output shape should be (batch_size, 77, 1024)
        self.assertEqual(
            output.shape, (1, 77, 1024), "Output shape is incorrect."
        )

        # Another test for multiple inputs
        output = self.embedder(self.text_inputs)
        self.assertEqual(
            output.shape, (2, 77, 1024), "Output shape is incorrect."
        )

    def test_precision_fp32(self):
        """Test that the precision is correctly set to fp32"""
        # Check that the output is of type float32
        output = self.embedder(self.text_input)
        self.assertEqual(
            output.dtype, torch.float32, "Output dtype is not float32."
        )

    def test_freeze_weights(self):
        """Test that the model weights are frozen"""
        for param in self.embedder.model.parameters():
            self.assertFalse(
                param.requires_grad, "Model parameters should be frozen."
            )

    def test_cuda_device(self):
        """Test if the model runs on CUDA if available"""
        if torch.cuda.is_available():
            self.assertEqual(
                self.embedder.device.type, "cuda", "Model should be on CUDA."
            )
        else:
            self.assertEqual(
                self.embedder.device.type, "cpu", "Model should be on CPU."
            )


if __name__ == "__main__":
    unittest.main()
