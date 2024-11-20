import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from .tal import TaskAlignedAssigner
from .utils import (
    compute_ciou,
    distance2xyxybbox,
    make_anchors,
    scale_xyxy,
    xyxybbox2distance,
)

# Author: @dasbd72 (Sao-Hsuan Lin)
# Reference: [RangeKing Yolov8 Architecture](https://mmyolo.readthedocs.io/zh-cn/dev/recommended_topics/algorithm_descriptions/yolov8_description.html)
# Reference: [UltraLytics Yolov8 Implementation](https://github.com/ultralytics/ultralytics)

# YOLOv8 Configurations
YOLO_V8_CONFIGS = {
    "n": {"depth_multiplier": 0.33, "width_multiplier": 0.25, "ratio": 2.0},
    "s": {"depth_multiplier": 0.33, "width_multiplier": 0.5, "ratio": 2.0},
    "m": {"depth_multiplier": 0.67, "width_multiplier": 0.75, "ratio": 1.5},
    "l": {"depth_multiplier": 1.0, "width_multiplier": 1.0, "ratio": 1.0},
    "x": {"depth_multiplier": 1.0, "width_multiplier": 1.25, "ratio": 1.0},
}


class Conv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0
    ):
        super(Conv, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        kernel_size = int(kernel_size)
        stride = int(stride)
        padding = int(padding)

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Swish activation

        # Apply Kaiming (He) initialization to the convolution layer
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="relu"
        )

        # Initialize BatchNorm weights
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SPPF, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        kernel_size = int(kernel_size)
        assert in_channels % 2 == 0

        hidden_channels = in_channels // 2  # typically set hidden channels
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.pool1 = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )  # Used 3 times since no learnable parameters
        self.conv2 = Conv(hidden_channels * 4, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        y1 = self.pool1(x)
        y2 = self.pool1(y1)
        y3 = self.pool1(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        """
        If shortcut is True, the input and output dimensions should be the same to add them together.

        :param in_channels: int
        :param out_channels: int
        :param shortcut: bool
        """
        super(Bottleneck, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        shortcut = bool(shortcut)
        assert out_channels % 2 == 0

        hidden_channels = out_channels // 2
        self.conv1 = Conv(
            in_channels, hidden_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = Conv(
            hidden_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor):
        y = self.conv2(self.conv1(x))
        if self.add:
            return x + y
        else:
            return y


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2, shortcut=True):
        super(C2f, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        num_blocks = int(num_blocks)
        shortcut = bool(shortcut)
        assert out_channels % 2 == 0

        hidden_channels = out_channels // 2
        self.conv1 = Conv(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.blocks = nn.ModuleList(
            [
                Bottleneck(hidden_channels, hidden_channels, shortcut)
                for _ in range(num_blocks)
            ]
        )
        self.conv2 = Conv(
            (num_blocks + 2) * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        upper, lower = torch.chunk(x, 2, dim=1)
        outputs = [upper, lower]
        for block in self.blocks:
            lower = block(lower)
            outputs.append(lower)
        return self.conv2(torch.cat(outputs, 1))


class Detect(nn.Module):
    def __init__(
        self,
        in_channels,
        bbox_hidden_channels,
        cls_hidden_channels,
        num_classes,
        reg_max=16,
    ):
        super(Detect, self).__init__()
        in_channels = int(in_channels)
        bbox_hidden_channels = int(bbox_hidden_channels)
        cls_hidden_channels = int(cls_hidden_channels)
        num_classes = int(num_classes)
        reg_max = int(reg_max)

        self.bbox_conv1 = Conv(
            in_channels,
            bbox_hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bbox_conv2 = Conv(
            bbox_hidden_channels,
            bbox_hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bbox_conv3 = nn.Conv2d(
            bbox_hidden_channels,
            reg_max * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.cls_conv1 = Conv(
            in_channels,
            cls_hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.cls_conv2 = Conv(
            cls_hidden_channels,
            cls_hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.cls_conv3 = nn.Conv2d(
            cls_hidden_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Apply Kaiming (He) initialization to the convolution layers
        nn.init.kaiming_normal_(
            self.bbox_conv3.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.cls_conv3.weight, mode="fan_out", nonlinearity="relu"
        )
        # Initialize biases to zero
        if self.bbox_conv3.bias is not None:
            nn.init.constant_(self.bbox_conv3.bias, 0)
        if self.cls_conv3.bias is not None:
            nn.init.constant_(self.cls_conv3.bias, 0)

    def forward(self, x: torch.Tensor):
        bbox = self.bbox_conv3(
            self.bbox_conv2(self.bbox_conv1(x))
        )  # (batch_size, reg_max*4, H, W)
        cls = self.cls_conv3(
            self.cls_conv2(self.cls_conv1(x))
        )  # (batch_size, num_classes, H, W)
        x = torch.cat(
            [bbox, cls], 1
        )  # (batch_size, reg_max*4 + num_classes, H, W)
        return x


class Backbone(nn.Module):
    def __init__(self, depth_multiplier=1.0, width_multiplier=1.0, ratio=1.0):
        super(Backbone, self).__init__()

        # Abbreviations
        d = depth_multiplier
        w = width_multiplier
        r = ratio

        self.conv1 = Conv(
            3, w * 64, kernel_size=3, stride=2, padding=1
        )  # 320x320
        self.conv2 = Conv(
            w * 64, w * 128, kernel_size=3, stride=2, padding=1
        )  # 160x160
        self.c2f1 = C2f(
            w * 128, w * 128, num_blocks=d * 3, shortcut=True
        )  # 160x160
        self.conv3 = Conv(
            w * 128, w * 256, kernel_size=3, stride=2, padding=1
        )  # 80x80
        self.c2f2 = C2f(
            w * 256, w * 256, num_blocks=d * 6, shortcut=True
        )  # 80x80
        self.conv4 = Conv(
            w * 256, w * 512, kernel_size=3, stride=2, padding=1
        )  # 40x40
        self.c2f3 = C2f(
            w * 512, w * 512, num_blocks=d * 6, shortcut=True
        )  # 40x40
        self.conv5 = Conv(
            w * 512, w * r * 512, kernel_size=3, stride=2, padding=1
        )  # 20x20
        self.c2f4 = C2f(
            w * r * 512, w * r * 512, num_blocks=d * 3, shortcut=True
        )  # 20x20

    def forward(self, x: torch.Tensor):
        assert x.shape[2] == x.shape[3] and x.shape[2] == 640
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c2f1(x)
        x = self.conv3(x)
        x1 = x = self.c2f2(x)
        x = self.conv4(x)
        x2 = x = self.c2f3(x)
        x = self.conv5(x)
        x = self.c2f4(x)
        return [x1, x2, x]


class Head(nn.Module):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        ratio=1.0,
        reg_max=16,
        num_classes=20,
    ):
        super(Head, self).__init__()

        # Abbreviations
        d = depth_multiplier
        w = width_multiplier
        r = ratio

        self.sppf = SPPF(w * r * 512, w * r * 512, kernel_size=5)  # 20x20
        self.up11 = nn.Upsample(scale_factor=2, mode="nearest")  # 40x40
        # Concat
        self.c2f11 = C2f(
            w * (1 + r) * 512,
            w * 512,
            num_blocks=d * 3,
            shortcut=False,
        )  # 40x40
        self.up22 = nn.Upsample(scale_factor=2, mode="nearest")  # 80x80
        # Concat
        self.c2f12 = C2f(
            w * 768, w * 256, num_blocks=d * 3, shortcut=False
        )  # 80x80

        self.conv21 = Conv(
            w * 256, w * 256, kernel_size=3, stride=2, padding=1
        )  # 40x40
        # Concat
        self.c2f21 = C2f(
            w * 768, w * 512, num_blocks=d * 3, shortcut=False
        )  # 40x40
        self.conv22 = Conv(
            w * 512, w * 512, kernel_size=3, stride=2, padding=1
        )  # 20x20
        # Concat
        self.c2f22 = C2f(
            w * (1 + r) * 512, w * r * 512, num_blocks=d * 3, shortcut=False
        )  # 20x20

        bbox_hidden_channels = max(16, w * r * 512 // 4, reg_max * 4)
        cls_hidden_channels = max(w * r * 512, min(num_classes, 100))
        self.detect1 = Detect(
            w * r * 512,
            bbox_hidden_channels,
            cls_hidden_channels,
            num_classes,
            reg_max=reg_max,
        )  # large objects
        self.detect2 = Detect(
            w * 512,
            bbox_hidden_channels,
            cls_hidden_channels,
            num_classes,
            reg_max=reg_max,
        )  # medium objects
        self.detect3 = Detect(
            w * 256,
            bbox_hidden_channels,
            cls_hidden_channels,
            num_classes,
            reg_max=reg_max,
        )  # small objects

    def forward(self, x_: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x_p3, x_p4, x_p5 = x_

        x_sppf = x = self.sppf(x_p5)
        x = self.up11(x)
        x_c2f11 = x = self.c2f11(torch.cat([x, x_p4], 1))
        x = self.up22(x)
        x_c2f12 = x = self.c2f12(torch.cat([x, x_p3], 1))

        x = self.conv21(x)
        x_c2f21 = x = self.c2f21(torch.cat([x, x_c2f11], 1))
        x = self.conv22(x)
        x = self.c2f22(torch.cat([x, x_sppf], 1))

        x_large = self.detect1(x)
        x_medium = self.detect2(x_c2f21)
        x_small = self.detect3(x_c2f12)
        return x_large, x_medium, x_small


class DFL(nn.Module):
    """
    DFL (Distribution Focal Loss).
    """

    def __init__(self, reg_max=16):
        super(DFL, self).__init__()
        self.reg_max = reg_max

        # Convolutional layer for weighted summation of bins
        self.conv = nn.Conv2d(
            reg_max, 1, kernel_size=1, stride=1, padding=0, bias=False
        ).requires_grad_(False)
        # Initialize the convolutional layer with a range of values
        self.conv.weight.data[:] = torch.arange(
            reg_max, dtype=torch.float32
        ).view(1, reg_max, 1, 1)

    def forward(self, x: torch.Tensor):
        """
        Apply softmax and weighted summation to decode reg_max outputs into box coordinates.

        :param x torch.Tensor: shape (batch_size, A, 4*reg_max).

        :rtype: torch.Tensor
        :return: Shape (batch_size, A, 4). Decoded bounding box coordinates.
        """
        batch_size, a, _ = x.shape
        x = x.view(
            batch_size, a, 4, self.reg_max
        )  # (batch_size, A, 4, reg_max)
        x = x.permute(0, 3, 1, 2)  # (batch_size, reg_max, A, 4)
        # Softmax on the reg_max dimension
        x = F.softmax(x, dim=1)  # (batch_size, reg_max, A, 4)
        # Apply the convolution to perform the weighted summation over the reg_max dimension
        x = self.conv(x)  # (batch_size, 1, A, 4)
        # Reshape to final output shape (batch_size, A, 4)
        x = x.view(batch_size, 4, a).permute(0, 2, 1)  # (batch_size, A, 4)
        return x


class YoloV8(nn.Module):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        ratio=1.0,
        num_classes=20,
        reg_max=16,
        pred_max=100,
    ):
        super(YoloV8, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.pred_max = pred_max

        self.strides = torch.tensor([32, 16, 8], dtype=torch.float32)

        self.backbone = Backbone(depth_multiplier, width_multiplier, ratio)
        self.head = Head(
            depth_multiplier, width_multiplier, ratio, reg_max, num_classes
        )
        self.dfl = DFL(reg_max=reg_max)

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: shape=(batch_size, 3, H, W). Input images.
        :rtype: tuple[torch.Tensor,torch.Tensor,torch.Tensor]
        :return: x_large, x_medium, x_small, shapes=(batch_size, reg_max*4 + num_classes, H, W), ...
        """
        x = self.backbone(x)
        xs = self.head(x)  # x_large, x_medium, x_small
        return xs

    def inference(self, x: torch.Tensor):
        """
        Inference the model.

        :param x torch.Tensor: shape=(batch_size, 3, H, W). Input images.
        :rtype: torch.Tensor
        :return: shape=(batch_size, min(pred_max, A), 6)
        """
        x = self.forward(x)
        x = self._postprocess(x)
        return x

    def freeze_backbone(self):
        # Dummy method to freeze the backbone
        pass

    def unfreeze_backbone(self):
        # Dummy method to unfreeze the backbone
        pass

    def _postprocess(
        self, xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Postprocess the output of the model.

        :param xs tuple[torch.Tensor,torch.Tensor,torch.Tensor]: x_large, x_medium, x_small.
            Shapes=(batch_size, reg_max*4 + num_classes, H, W), ...
        :rtype: torch.Tensor
        :return: shape=(batch_size, min(pred_max, A), 6)
            The last dimension contains (x1, y1, x2, y2, class, score).
        """
        anchors, strides = make_anchors(
            xs, self.strides, grid_cell_offset=0.5
        )  # (A, 2), (A, 1), where A = Hl*Wl + Hm*Wm + Hs*Ws

        # Step 1: Process the outputs into bounding boxes and class predictions
        # Concatenate the outputs and reshape
        xs = [x.view(x.shape[0], x.shape[1], -1) for x in xs]
        x = torch.cat(xs, 2)  # (batch_size, reg_max*4 + num_classes, A)
        x = x.permute(0, 2, 1)  # (batch_size, A, reg_max*4 + num_classes)

        # Get batch size and number of anchors
        batch_size, num_anchors, _ = x.shape

        # Separate the bounding box and class predictions
        box_pred, cls_pred = x.split(
            [self.reg_max * 4, self.num_classes], dim=-1
        )  # (batch_size, A, reg_max*4), (batch_size, A, num_classes)

        # Decode the bounding boxes and scale them to the target image size
        box_pred = self.dfl(box_pred)  # (batch_size, A, 4)
        box_pred = self._decode_bboxes(
            box_pred, anchors.unsqueeze(0)
        ) * strides.unsqueeze(
            0
        )  # (batch_size, A, 4), unsqueeze first dimension to match batch_size

        # Apply sigmoid to the class predictions
        cls_pred = cls_pred.sigmoid()  # (batch_size, A, num_classes)
        # x = torch.cat([box_pred, cls_pred], dim=-1)

        # Step 2: Postprocess the predictions and select the top-k predictions
        # Select the top-k predictions
        indices = cls_pred.amax(dim=-1)  # (batch_size, A)
        _, indices = indices.topk(
            min(self.pred_max, num_anchors)
        )  # values (batch_size, min(pred_max, A)), indices (batch_size, min(pred_max, A))
        indices = indices.unsqueeze(-1)  # (batch_size, min(pred_max, A), 1)

        # Gather the top-k predictions
        box_pred = box_pred.gather(
            dim=1, index=indices.expand(-1, -1, 4)
        )  # (batch_size, min(pred_max, A), 4)
        score_pred = cls_pred.gather(
            dim=1, index=indices.expand(-1, -1, self.num_classes)
        )  # (batch_size, min(pred_max, A), num_classes)

        score_pred, indices = score_pred.flatten(1).topk(
            min(self.pred_max, num_anchors)
        )  # (batch_size, min(pred_max, A)), (batch_size, min(pred_max, A))
        i = torch.arange(batch_size, device=x.device).unsqueeze(
            1
        )  # (batch_size, 1)
        x = torch.cat(
            [
                box_pred[i, indices // self.num_classes],
                (indices % self.num_classes)[..., None].float(),
                score_pred[..., None],
            ],
            dim=-1,
        )  # (batch_size, min(pred_max, A), 6), (x1, y1, x2, y2, class, score)
        return x

    def _decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor):
        """
        Decode the predicted bounding boxes.

        :param bboxes torch.Tensor: shape=(batch_size, A, 4)
        :param anchors torch.Tensor: shape=(1, A, 2)
        :rtype: torch.Tensor
        :return: shape=(batch_size, A, 4)
        """
        bboxes = distance2xyxybbox(bboxes, anchors, dim=-1)
        return bboxes

    @staticmethod
    def get_yolo_v8_n(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8(
            depth_multiplier=YOLO_V8_CONFIGS["n"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["n"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["n"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_s(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8(
            depth_multiplier=YOLO_V8_CONFIGS["s"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["s"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["s"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_m(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8(
            depth_multiplier=YOLO_V8_CONFIGS["m"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["m"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["m"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_l(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8(
            depth_multiplier=YOLO_V8_CONFIGS["l"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["l"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["l"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_x(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8(
            depth_multiplier=YOLO_V8_CONFIGS["x"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["x"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["x"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        ratio=1.0,
        freeze=True,
    ):
        super(ResNetBackbone, self).__init__()
        # Abbreviations
        w = width_multiplier
        r = ratio

        # Load a pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Extract layers for different scales (C3, C4, C5)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        # Map ResNet output channels to desired output channels
        self.reduce_c3 = Conv(512, w * 256, kernel_size=1)
        self.reduce_c4 = Conv(1024, w * 512, kernel_size=1)
        self.reduce_c5 = Conv(2048, w * r * 512, kernel_size=1)

        # Optionally freeze ResNet weights
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Forward pass through ResNet-50 layers
        x = self.stem(x)  # Output 1/4 size
        c3 = self.layer2(self.layer1(x))  # Output 1/8 size
        c4 = self.layer3(c3)  # Output 1/16 size
        c5 = self.layer4(c4)  # Output 1/32 size

        # Reduce channels to match YOLOv8 expectations
        c3 = self.reduce_c3(c3)  # (batch, w * 256, h/8, w/8)
        c4 = self.reduce_c4(c4)  # (batch, w * 512, h/16, w/16)
        c5 = self.reduce_c5(c5)  # (batch, w * r * 512, h/32, w/32)

        return c3, c4, c5


class YoloV8WithResNet(YoloV8):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        ratio=1.0,
        num_classes=20,
        reg_max=16,
        pred_max=100,
    ):
        super(YoloV8WithResNet, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.pred_max = pred_max

        self.strides = torch.tensor([32, 16, 8], dtype=torch.float32)

        # Replace Backbone with ResNet-based backbone
        self.backbone = ResNetBackbone(
            depth_multiplier, width_multiplier, ratio
        )
        self.head = Head(
            depth_multiplier, width_multiplier, ratio, reg_max, num_classes
        )
        self.dfl = DFL(reg_max=reg_max)

    def forward(self, x):
        x = self.backbone(x)  # ResNet outputs
        xs = self.head(x)  # x_large, x_medium, x_small
        return xs

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    @staticmethod
    def get_yolo_v8_n(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithResNet(
            depth_multiplier=YOLO_V8_CONFIGS["n"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["n"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["n"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_s(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithResNet(
            depth_multiplier=YOLO_V8_CONFIGS["s"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["s"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["s"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_m(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithResNet(
            depth_multiplier=YOLO_V8_CONFIGS["m"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["m"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["m"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_l(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithResNet(
            depth_multiplier=YOLO_V8_CONFIGS["l"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["l"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["l"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_x(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithResNet(
            depth_multiplier=YOLO_V8_CONFIGS["x"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["x"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["x"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )


class EfficientNetB3Backbone(nn.Module):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        ratio=1.0,
        freeze=True,
    ):
        super(EfficientNetB3Backbone, self).__init__()
        # Abbreviations
        w = width_multiplier
        r = ratio

        # Load a pre-trained EfficientNet-B3 model
        efficientnet = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT
        )

        # Extract layers for different scales
        self.stem = nn.Sequential(
            efficientnet.features[
                0
            ],  # Initial convolution layer (batch, 40, h/2, w/2)
            efficientnet.features[1],  # First block (batch, 24, h/2, w/2)
        )
        self.layer1 = nn.Sequential(
            *efficientnet.features[2]
        )  # C2 (batch, 40, h/4, w/4)
        self.layer2 = nn.Sequential(
            *efficientnet.features[3]
        )  # C3 (batch, 48, h/8, w/8)
        self.layer3 = nn.Sequential(
            *efficientnet.features[4:6]
        )  # C4 (batch, 136, h/16, w/16)
        self.layer4 = efficientnet.features[6:8]  # C5 (batch, 384, h/32, w/32)

        # Map EfficientNet output channels to desired output channels
        self.reduce_c3 = nn.Conv2d(48, int(w * 256), kernel_size=1)
        self.reduce_c4 = nn.Conv2d(136, int(w * 512), kernel_size=1)
        self.reduce_c5 = nn.Conv2d(384, int(w * r * 512), kernel_size=1)

        # Optionally freeze EfficientNet weights
        if freeze:
            for param in efficientnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Forward pass through EfficientNet-B3 layers
        x = self.stem(x)  # Output 1/4 size
        c3 = self.layer2(self.layer1(x))  # Output 1/8 size
        c4 = self.layer3(c3)  # Output 1/16 size
        c5 = self.layer4(c4)  # Output 1/32 size

        # Increace channels to match desired output expectations
        c3 = self.reduce_c3(c3)  # (batch, w * 256, h/8, w/8)
        c4 = self.reduce_c4(c4)  # (batch, w * 512, h/16, w/16)
        c5 = self.reduce_c5(c5)  # (batch, w * r * 512, h/32, w/32)

        return c3, c4, c5


class YoloV8WithEfficientNetB3(YoloV8):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        ratio=1.0,
        num_classes=20,
        reg_max=16,
        pred_max=100,
    ):
        super(YoloV8WithEfficientNetB3, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.pred_max = pred_max

        self.strides = torch.tensor([32, 16, 8], dtype=torch.float32)

        # Replace Backbone with EfficientNet-based backbone
        self.backbone = EfficientNetB3Backbone(
            depth_multiplier, width_multiplier, ratio
        )
        self.head = Head(
            depth_multiplier, width_multiplier, ratio, reg_max, num_classes
        )
        self.dfl = DFL(reg_max=reg_max)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    @staticmethod
    def get_yolo_v8_n(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithEfficientNetB3(
            depth_multiplier=YOLO_V8_CONFIGS["n"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["n"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["n"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_s(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithEfficientNetB3(
            depth_multiplier=YOLO_V8_CONFIGS["s"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["s"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["s"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_m(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithEfficientNetB3(
            depth_multiplier=YOLO_V8_CONFIGS["m"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["m"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["m"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_l(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithEfficientNetB3(
            depth_multiplier=YOLO_V8_CONFIGS["l"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["l"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["l"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_x(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithEfficientNetB3(
            depth_multiplier=YOLO_V8_CONFIGS["x"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["x"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["x"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )


class DenseNet121Backbone(nn.Module):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        ratio=1.0,
        freeze=True,
    ):
        super(DenseNet121Backbone, self).__init__()
        # Abbreviations
        w = width_multiplier
        r = ratio

        # Load a pre-trained DenseNet-121 model
        densenet = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        # Extract layers for different scales
        self.stem = nn.Sequential(
            densenet.features.conv0,  # Initial convolution (batch, 64, h/2, w/2)
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
        )  # Output 1/4 size

        self.layer1 = (
            densenet.features.denseblock1
        )  # C2 (batch, 256, h/4, w/4)
        self.transition1 = densenet.features.transition1  # Downsample
        self.layer2 = (
            densenet.features.denseblock2
        )  # C3 (batch, 512, h/8, w/8)

        self.transition2 = densenet.features.transition2  # Downsample
        self.layer3 = (
            densenet.features.denseblock3
        )  # C4 (batch, 1024, h/16, w/16)

        self.transition3 = densenet.features.transition3  # Downsample
        self.layer4 = (
            densenet.features.denseblock4
        )  # C5 (batch, 1024, h/32, w/32)

        # Map DenseNet output channels to desired output channels
        self.reduce_c3 = nn.Conv2d(512, int(w * 256), kernel_size=1)
        self.reduce_c4 = nn.Conv2d(1024, int(w * 512), kernel_size=1)
        self.reduce_c5 = nn.Conv2d(1024, int(w * r * 512), kernel_size=1)

        # Optionally freeze DenseNet weights
        if freeze:
            for param in densenet.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Forward pass through DenseNet121 layers
        x = self.stem(x)  # Output 1/4 size
        c2 = self.layer1(x)
        c3 = self.layer2(self.transition1(c2))  # Output 1/8 size
        c4 = self.layer3(self.transition2(c3))  # Output 1/16 size
        c5 = self.layer4(self.transition3(c4))  # Output 1/32 size

        # Reduce channels to match desired output expectations
        c3 = self.reduce_c3(c3)  # (batch, w * 256, h/8, w/8)
        c4 = self.reduce_c4(c4)  # (batch, w * 512, h/16, w/16)
        c5 = self.reduce_c5(c5)  # (batch, w * r * 512, h/32, w/32)

        return c3, c4, c5


class YoloV8WithDenseNet121(YoloV8):
    def __init__(
        self,
        depth_multiplier=1.0,
        width_multiplier=1.0,
        ratio=1.0,
        num_classes=20,
        reg_max=16,
        pred_max=100,
    ):
        super(YoloV8WithDenseNet121, self).__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.pred_max = pred_max

        self.strides = torch.tensor([32, 16, 8], dtype=torch.float32)

        # Replace Backbone with DenseNet-based backbone
        self.backbone = DenseNet121Backbone(
            depth_multiplier, width_multiplier, ratio
        )
        self.head = Head(
            depth_multiplier, width_multiplier, ratio, reg_max, num_classes
        )
        self.dfl = DFL(reg_max=reg_max)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    @staticmethod
    def get_yolo_v8_n(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithDenseNet121(
            depth_multiplier=YOLO_V8_CONFIGS["n"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["n"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["n"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_s(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithDenseNet121(
            depth_multiplier=YOLO_V8_CONFIGS["s"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["s"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["s"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_m(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithDenseNet121(
            depth_multiplier=YOLO_V8_CONFIGS["m"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["m"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["m"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_l(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithDenseNet121(
            depth_multiplier=YOLO_V8_CONFIGS["l"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["l"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["l"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )

    @staticmethod
    def get_yolo_v8_x(num_classes=20, reg_max=16, pred_max=100):
        return YoloV8WithDenseNet121(
            depth_multiplier=YOLO_V8_CONFIGS["x"]["depth_multiplier"],
            width_multiplier=YOLO_V8_CONFIGS["x"]["width_multiplier"],
            ratio=YOLO_V8_CONFIGS["x"]["ratio"],
            num_classes=num_classes,
            reg_max=reg_max,
            pred_max=pred_max,
        )


class CIoULoss(nn.Module):
    """
    Compute the CIoU (Complete Intersection over Union) loss.
    """

    def __init__(self, eps=1e-7):
        super(CIoULoss, self).__init__()
        self.eps = eps

    def forward(self, box_preds: torch.Tensor, box_labels: torch.Tensor):
        """
        Compute the Complete Intersection over Union (CIoU) loss of two set of boxes.

        $$
        CIoU = IoU - (center_distance / enclose_diagonal) - aspect_ratio_consistency
        $$
        $$
        CIoULoss = 1 - CIoU
        $$

        :param box_preds torch.Tensor: shape=(batch_size, A, 4). Predicted bounding boxes.
        :param box_labels torch.Tensor: shape=(batch_size, A, 4,). Target bounding boxes.

        :rtype: torch.Tensor
        :return: shape=(batch_size, N). CIoU loss.
        """
        ciou = compute_ciou(
            box_preds, box_labels, eps=self.eps
        )  # batch_size, N
        ciou_loss = 1 - ciou
        return ciou_loss


class DFLoss(nn.Module):
    """
    Distribution Focal Loss.
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, dist_pred: torch.Tensor, dist_labels: torch.Tensor):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391

        :param dist_pred torch.Tensor: shape=(A, reg_max). Predicted distribution.
        :param dist_labels torch.Tensor: shape=(A, 4). Target distribution.

        :rtype: torch.Tensor
        :return: shape=(). Loss.
        """
        dist_labels = dist_labels.clamp(min=0, max=self.reg_max - 1 - 0.01)
        tl = dist_labels.floor().long()  # target left
        tr = tl + 1  # target right
        wl = tr - dist_labels  # weight left
        wr = 1 - wl  # weight right
        loss = (
            F.cross_entropy(
                dist_pred,
                tl.view(-1),
                reduction="none",
            ).view(tl.shape)
            * wl
            + F.cross_entropy(
                dist_pred,
                tr.view(-1),
                reduction="none",
            ).view(tr.shape)
            * wr
        ).mean(-1, keepdim=True)
        return loss


class YoloV8Loss(nn.Module):
    """
    YoloV8 loss function.

    Uses TaskAlignedAssigner from ultralytics to assign the boxes to the predictions.

    [TaskAlignedAssigner](https://docs.ultralytics.com/reference/utils/tal/#ultralytics.utils.tal.TaskAlignedAssigner)
    """

    def __init__(
        self,
        model: YoloV8,
        box_loss_weight=7.5,
        cls_loss_weight=0.5,
        dfl_loss_weight=1.5,
        eps=1e-9,
    ):
        super(YoloV8Loss, self).__init__()
        self.num_classes = model.num_classes
        self.reg_max = model.reg_max
        self.pred_max = model.pred_max
        self.strides = model.strides
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.eps = eps

        self.dfl = DFL(reg_max=self.reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ciou_loss = CIoULoss(eps=self.eps)
        self.dfl_loss = DFLoss(reg_max=self.reg_max)
        self.assigner = TaskAlignedAssigner(
            num_classes=self.num_classes, eps=self.eps, alpha=0.5, beta=6.0
        )

        self._prev_loss = None

    def __call__(
        self,
        predicts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
    ):
        """
        Compute the YoloV8 loss.

        :param predicts tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predictions of the model,
            shapes=(batch_size, reg_max*4 + num_classes, H1, W1), (batch_size, reg_max*4 + num_classes, H2, W2), (batch_size, reg_max*4 + num_classes, H3, W3).
        :param labels torch.Tensor: Labels of the model.
            shape=(num_labels, 6), each row contains (batch_idx, x1, y1, x2, y2, class), with num_labels being the number of truth bounding boxes in the batch.
            Will be converted to shape=(batch_size, max_n_box, 5) where max_n_box is the maximum number of counts of identical batch_idx.

        :rtype: torch.Tensor
        :return: Loss.
        """
        dtype = predicts[0].dtype
        device = predicts[0].device
        batch_size = predicts[0].shape[0]
        image_height, image_width = predicts[0].shape[2] * int(
            self.strides[0]
        ), predicts[0].shape[3] * int(self.strides[0])
        assert image_height == image_width and image_height == 640
        # num_labels = labels.shape[0]

        # Predictions
        anchors, strides = make_anchors(
            predicts, self.strides, grid_cell_offset=0.5
        )  # (A, 2), (A, 1)

        preds = [
            x.view(batch_size, self.reg_max * 4 + self.num_classes, -1)
            for x in predicts
        ]  # (batch_size, reg_max*4 + num_classes, A1), (batch_size, reg_max*4 + num_classes, A2), (batch_size, reg_max*4 + num_classes, A3)
        pred = torch.cat(preds, 2)  # (batch_size, reg_max*4 + num_classes, A)
        pred = pred.permute(
            0, 2, 1
        ).contiguous()  # (batch_size, A, reg_max*4 + num_classes)
        distri_pred, scores_pred = pred.split(
            [self.reg_max * 4, self.num_classes], dim=2
        )  # (batch_size, A, reg_max*4), (batch_size, A, num_classes)
        box_pred = self.dfl(distri_pred)  # (batch_size, A, 4)
        box_pred = self._decode_bboxes(
            box_pred, anchors.unsqueeze(0)
        ) * strides.unsqueeze(
            0
        )  # (batch_size, A, 4)

        # Labels
        box_labels, scores_labels, labels_mask = self._process_labels(
            labels, batch_size
        )  # (batch_size, max_n_box, 4), (batch_size, max_n_box, 1), (batch_size, max_n_box, 1)

        # Before computing loss, we need to match the boxes and labels
        box_labels, scores_labels, labels_mask = self._assign(
            box_pred.detach(),
            scores_pred.detach().sigmoid(),
            box_labels,
            scores_labels,
            labels_mask,
            anchors * strides,
        )  # (batch_size, A, 4), (batch_size, A, num_classes), (batch_size, A, 1)
        scores_sum_labels = max(scores_labels.sum(), 1)

        cls_loss = (
            self.bce(scores_pred, scores_labels.to(dtype=dtype)).sum()
            / scores_sum_labels
        )
        if labels_mask.sum() > 0:
            box_pred /= strides
            box_labels /= strides
            weight = scores_labels.sum(dim=-1)[
                labels_mask
            ]  # (num_boxes_filtered,)
            box_loss = (
                self.ciou_loss(box_pred[labels_mask], box_labels[labels_mask])
                * weight  # (num_boxes_filtered,)
            ).sum() / scores_sum_labels
            box_loss *= self.box_loss_weight
            distri_labels = xyxybbox2distance(
                box_labels, anchors, self.reg_max, dim=-1
            )  # (batch_size, A, 4*reg_max)
            dfl_loss = (
                self.dfl_loss(
                    distri_pred[labels_mask].view(-1, self.reg_max),
                    distri_labels[labels_mask],
                )
                * weight
            ).sum() / scores_sum_labels
        else:
            print("Warning: No boxes are matched.")

        loss = torch.zeros(3, device=device)  # box, cls, dfl
        loss[0] = box_loss * self.box_loss_weight * batch_size
        loss[1] = cls_loss * self.cls_loss_weight * batch_size
        loss[2] = dfl_loss * self.dfl_loss_weight * batch_size

        self._prev_loss = loss.detach()
        return loss.sum()

    def _decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor):
        """
        Decode the predicted bounding boxes.
        """
        bboxes = distance2xyxybbox(bboxes, anchors, dim=-1)
        return bboxes

    def _process_labels(self, labels: torch.Tensor, batch_size: int):
        device = labels.device
        batch_idx = labels[:, 0].long()  # (num_labels,)
        _, batch_idx_counts = batch_idx.unique(
            return_counts=True
        )  # (batch_size, 1)
        max_n_box = int(batch_idx_counts.max())
        wide_labels = torch.zeros(
            batch_size, max_n_box, 5, device=device
        )  # (batch_size, max_n_box, 5)

        # Fill the wide_labels tensor
        for i in range(batch_size):
            matched = batch_idx == i
            wide_labels[i, : matched.sum(), :] = labels[matched, 1:]

        box_labels, cls_labels = wide_labels.split(
            [4, 1], dim=-1
        )  # (batch_size, max_n_box, 4), (batch_size, max_n_box, 1)
        box_labels = scale_xyxy(box_labels, 640, dim=-1)

        # Compute mask of avtived boxes
        labels_mask = (
            box_labels.sum(-1, keepdim=True) > 0
        )  # (batch_size, max_n_box, 1)
        return box_labels, cls_labels, labels_mask

    @torch.no_grad()
    def _assign(
        self,
        box_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        box_labels: torch.Tensor,
        cls_labels: torch.Tensor,
        labels_mask: torch.Tensor,
        anchors: torch.Tensor,
    ):
        """
        Task-aligned One-stage Object Detection

        :param box_pred torch.Tensor: shape=(batch_size, A, 4). Predicted bounding boxes.
        :param cls_pred torch.Tensor: shape=(batch_size, A, num_classes). Predicted class scores.
        :param box_labels torch.Tensor: shape=(batch_size, max_n_box, 4). Ground-truth bounding boxes.
        :param cls_labels torch.Tensor: shape=(batch_size, max_n_box, 1). Ground-truth class labels.
        :param labels_mask torch.Tensor: shape=(batch_size, max_n_box, 1). Mask of activated boxes.
        :param anchors torch.Tensor: shape=(A, 2). Anchors.

        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :return: box_labels, cls_labels, labels_mask.
            Shapes=(batch_size, A, 4), (batch_size, A, num_classes), (batch_size, A, 1).
        """
        _, box_labels, cls_labels, labels_mask, _ = self.assigner(
            cls_pred, box_pred, anchors, cls_labels, box_labels, labels_mask
        )
        return box_labels, cls_labels, labels_mask
