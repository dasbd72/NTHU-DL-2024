import unittest

import torch

from .layers import (
    DFL,
    CIoULoss,
    YoloV8,
    YoloV8Loss,
    YoloV8WithDenseNet121,
    YoloV8WithEfficientNetB3,
    YoloV8WithResNet,
)


class BaseTestYoloV8(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_classes = 20
        self.reg_max = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_input = torch.randn(self.batch_size, 3, 640, 640).to(
            self.device
        )

    def _pred_shape(self, h, w):
        return (self.batch_size, self.reg_max * 4 + self.num_classes, h, w)

    def _assert_output(
        self, output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        self.assertEqual(output[0].shape, self._pred_shape(20, 20))
        self.assertEqual(output[1].shape, self._pred_shape(40, 40))
        self.assertEqual(output[2].shape, self._pred_shape(80, 80))

    def _test_one_model(self, model: YoloV8):
        self.assertIsNotNone(model)
        model.train()
        output = model(self.sample_input)
        self._assert_output(output)


class TestYoloV8(BaseTestYoloV8):
    def test_yolo_v8_n(self):
        model = YoloV8.get_yolo_v8_n(
            num_classes=self.num_classes, reg_max=self.reg_max
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_s(self):
        model = YoloV8.get_yolo_v8_s(
            num_classes=self.num_classes, reg_max=self.reg_max
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_m(self):
        model = YoloV8.get_yolo_v8_m(
            num_classes=self.num_classes, reg_max=self.reg_max
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_l(self):
        model = YoloV8.get_yolo_v8_l(
            num_classes=self.num_classes, reg_max=self.reg_max
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_x(self):
        model = YoloV8.get_yolo_v8_x(
            num_classes=self.num_classes, reg_max=self.reg_max
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_with_resnet(self):
        model = YoloV8WithResNet.get_yolo_v8_n(
            num_classes=self.num_classes, reg_max=self.reg_max
        ).to(self.device)
        self._test_one_model(model)
        model.freeze_backbone()
        model.unfreeze_backbone()

    def test_yolo_v8_with_resnet(self):
        model = YoloV8WithResNet.get_yolo_v8_x(
            num_classes=self.num_classes, reg_max=self.reg_max
        ).to(self.device)
        self._test_one_model(model)


class TestYoloV8B3(BaseTestYoloV8):
    def test_yolo_v8_with_b3_n(self):
        model = YoloV8WithEfficientNetB3.get_yolo_v8_n(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_with_b3_m(self):
        model = YoloV8WithEfficientNetB3.get_yolo_v8_m(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
        ).to(self.device)
        self._test_one_model(model)


class TestYoloV8DN121(BaseTestYoloV8):
    def test_yolo_v8_with_dn121_n(self):
        model = YoloV8WithDenseNet121.get_yolo_v8_n(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_with_dn121_m(self):
        model = YoloV8WithDenseNet121.get_yolo_v8_m(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
        ).to(self.device)
        self._test_one_model(model)


class BaseTestYoloV8Inference(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_classes = 20
        self.reg_max = 10
        self.pred_max = 100
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_input = torch.randn(self.batch_size, 3, 640, 640).to(
            self.device
        )

    def _pred_shape(self):
        return (self.batch_size, self.pred_max, 6)

    def _assert_output(
        self, output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        self.assertEqual(output.shape, self._pred_shape())

    def _test_one_model(self, model: YoloV8):
        self.assertIsNotNone(model)
        with torch.no_grad():
            model.eval()
            output = model.inference(self.sample_input)
        self._assert_output(output)


class TestYoloV8Inference(BaseTestYoloV8Inference):
    def test_yolo_v8_n(self):
        model = YoloV8.get_yolo_v8_n(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            pred_max=self.pred_max,
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_x(self):
        model = YoloV8.get_yolo_v8_x(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            pred_max=self.pred_max,
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_with_resnet_n(self):
        model = YoloV8WithResNet.get_yolo_v8_n(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            pred_max=self.pred_max,
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_with_resnet_x(self):
        model = YoloV8WithResNet.get_yolo_v8_x(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            pred_max=self.pred_max,
        ).to(self.device)
        self._test_one_model(model)


class TestYoloV8B3Inference(BaseTestYoloV8Inference):
    def test_yolo_v8_with_b3_n(self):
        model = YoloV8WithEfficientNetB3.get_yolo_v8_n(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            pred_max=self.pred_max,
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_with_b3_m(self):
        model = YoloV8WithEfficientNetB3.get_yolo_v8_m(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            pred_max=self.pred_max,
        ).to(self.device)
        self._test_one_model(model)


class TestYoloV8DN121Inference(BaseTestYoloV8Inference):
    def test_yolo_v8_with_dn121_n(self):
        model = YoloV8WithDenseNet121.get_yolo_v8_n(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            pred_max=self.pred_max,
        ).to(self.device)
        self._test_one_model(model)

    def test_yolo_v8_with_dn121_m(self):
        model = YoloV8WithDenseNet121.get_yolo_v8_m(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            pred_max=self.pred_max,
        ).to(self.device)
        self._test_one_model(model)


class TestDFL(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.reg_max = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_input = torch.randn(
            self.batch_size, 400, self.reg_max * 4
        ).to(self.device)

    def test_dfl(self):
        model = DFL(reg_max=self.reg_max).to(self.device)
        self.assertIsNotNone(model)
        output = model(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, 400, 4))


class TestCIoULoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_boxes = 6400
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Random seed
        torch.manual_seed(0)

    def _sample_boxes(self, num_boxes=1):
        # Sample input should have shape=(batch_size, N, 4), and each boxes are in (x1, y1, x2, y2) format if num_boxes > 1
        # Sample target should have shape=(batch_size, 4), and each boxes are in (x1, y1, x2, y2) format if num_boxes == 1
        x1 = torch.randint(0, 639, (self.batch_size, num_boxes))
        y1 = torch.randint(0, 639, (self.batch_size, num_boxes))
        w = torch.randint(0, 640, (self.batch_size, num_boxes))
        h = torch.randint(0, 640, (self.batch_size, num_boxes))
        x2 = (x1 + w).clamp(max=640)
        y2 = (y1 + h).clamp(max=640)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def test_ciou_loss_1(self):
        loss = CIoULoss().to(self.device)
        self.assertIsNotNone(loss)
        sample_input = self._sample_boxes(num_boxes=1).to(self.device)
        sample_target = self._sample_boxes(num_boxes=1).to(self.device)
        output = loss(sample_input, sample_target)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_ciou_loss(self):
        loss = CIoULoss().to(self.device)
        self.assertIsNotNone(loss)
        sample_input = self._sample_boxes(num_boxes=6400).to(self.device)
        sample_target = self._sample_boxes(num_boxes=6400).to(self.device)
        output = loss(sample_input, sample_target)
        self.assertEqual(output.shape, (self.batch_size, 6400))


class TestYoloV8Loss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.max_n_box = 3
        self.num_classes = 20
        self.reg_max = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_input = torch.randn(self.batch_size, 3, 640, 640).to(
            self.device
        )
        sample_idx = (
            torch.arange(self.batch_size).repeat(self.max_n_box).unsqueeze(-1)
        )
        x1, y1, w, h = (
            torch.randint(0, 639, (self.batch_size * self.max_n_box, 1)),
            torch.randint(0, 639, (self.batch_size * self.max_n_box, 1)),
            torch.randint(0, 640, (self.batch_size * self.max_n_box, 1)),
            torch.randint(0, 640, (self.batch_size * self.max_n_box, 1)),
        )
        x2, y2 = (x1 + w).clamp(max=640), (y1 + h).clamp(max=640)
        sample_cls = torch.randint(
            0, self.num_classes, (self.batch_size * self.max_n_box, 1)
        )
        self.sample_labels = torch.cat(
            [sample_idx, x1, y1, x2, y2, sample_cls], dim=-1
        )
        self.sample_labels[:, 1:5] = self.sample_labels[:, 1:5] / 640
        self.sample_labels = self.sample_labels.to(self.device)

    def test_yolo_v8_loss(self):
        model = YoloV8.get_yolo_v8_m(
            num_classes=self.num_classes, reg_max=self.reg_max
        ).to(self.device)
        self.assertIsNotNone(model)
        yolo_loss = YoloV8Loss(model).to(self.device)
        self.assertIsNotNone(yolo_loss)
        output = model(self.sample_input)
        yolo_loss(output, self.sample_labels)


if __name__ == "__main__":
    unittest.main()
