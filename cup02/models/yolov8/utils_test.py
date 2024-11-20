import unittest

import torch

from .utils import (
    compute_ciou,
    distance2xyxybbox,
    make_anchors,
    scale_xyxy,
    xyxybbox2distance,
)


class TestUtilsFunctions(unittest.TestCase):
    """
    GPT Genrated Test Cases for Utils Functions
    """

    def setUp(self):
        self.batch_size = 4
        self.max_n_box = 3
        self.num_classes = 20
        self.reg_max = 16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.xs = (
            torch.randn(self.batch_size, 3, 40, 40).to(self.device),
            torch.randn(self.batch_size, 3, 20, 20).to(self.device),
            torch.randn(self.batch_size, 3, 10, 10).to(self.device),
        )
        self.strides = torch.tensor(
            [32, 16, 8], dtype=torch.int32, device=self.device
        )
        self.sample_boxes = torch.tensor(
            [[[0.1, 0.1, 0.5, 0.5]], [[0.3, 0.3, 0.7, 0.7]]],
            dtype=torch.float32,
            device=self.device,
        )
        self.sample_anchors = torch.tensor(
            [[0.25, 0.25]], dtype=torch.float32, device=self.device
        )

    def test_make_anchors(self):
        anchor_points, stride_tensor = make_anchors(self.xs, self.strides)
        self.assertEqual(anchor_points.shape[1], 2)  # Should have (N, 2) shape
        self.assertEqual(stride_tensor.shape[1], 1)  # Should have (N, 1) shape

    def test_distance2xyxybbox(self):
        distances = torch.randn(self.batch_size, self.max_n_box, 4).to(
            self.device
        )
        result = distance2xyxybbox(distances, self.sample_anchors)
        self.assertEqual(result.shape, (self.batch_size, self.max_n_box, 4))
        reverse = xyxybbox2distance(
            result, self.sample_anchors, reg_max=self.reg_max
        )
        self.assertTrue(torch.allclose(distances.clamp(min=0), reverse))

    def test_xyxybbox2distance(self):
        result = xyxybbox2distance(
            self.sample_boxes, self.sample_anchors, reg_max=self.reg_max
        )
        self.assertEqual(
            result.shape,
            (self.sample_boxes.shape[0], self.sample_boxes.shape[1], 4),
        )
        self.assertTrue((result >= 0).all())
        self.assertTrue((result < self.reg_max).all())

    def test_compute_ciou(self):
        box_preds = torch.tensor(
            [[[0.2, 0.2, 0.6, 0.6]], [[0.4, 0.4, 0.8, 0.8]]],
            dtype=torch.float32,
            device=self.device,
        )
        box_labels = torch.tensor(
            [[[0.1, 0.1, 0.5, 0.5]], [[0.3, 0.3, 0.7, 0.7]]],
            dtype=torch.float32,
            device=self.device,
        )
        result = compute_ciou(box_preds, box_labels)
        self.assertEqual(
            result.shape, (box_preds.shape[0], box_preds.shape[1])
        )
        self.assertTrue((result <= 1).all() and (result >= -1).all())

    def test_scale_xyxy(self):
        scale_factor = 1.5
        scaled_boxes = scale_xyxy(self.sample_boxes, scale=scale_factor)
        self.assertEqual(scaled_boxes.shape, self.sample_boxes.shape)
        sample_h, sample_w = (
            self.sample_boxes[:, :, 3] - self.sample_boxes[:, :, 1],
            self.sample_boxes[:, :, 2] - self.sample_boxes[:, :, 0],
        )
        scaled_h, scaled_w = (
            scaled_boxes[:, :, 3] - scaled_boxes[:, :, 1],
            scaled_boxes[:, :, 2] - scaled_boxes[:, :, 0],
        )
        self.assertTrue(
            torch.allclose(scaled_h, sample_h * scale_factor)
            and torch.allclose(scaled_w, sample_w * scale_factor)
        )


if __name__ == "__main__":
    unittest.main()
