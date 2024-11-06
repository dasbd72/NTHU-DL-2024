import torch
import torch.nn.functional as F
from torch import nn


class ConvLeakyReLU(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding
    ):
        super(ConvLeakyReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.relu = nn.LeakyReLU(0.1)

        # Kaiming initialization for Leaky ReLU
        nn.init.kaiming_normal_(
            self.conv.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu"
        )
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.relu(self.conv(x))


class YoloLoss(nn.Module):
    def __init__(
        self,
        cell_size,
        num_classes,
        boxes_per_cell,
        image_size,
        class_scale,
        object_scale,
        noobject_scale,
        coord_scale,
        device,
    ):
        super(YoloLoss, self).__init__()
        self.cell_size = cell_size
        self.num_classes = num_classes
        self.boxes_per_cell = boxes_per_cell
        self.image_size = image_size
        self.class_scale = class_scale
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.coord_scale = coord_scale
        self.device = device

        # Define base_boxes and scale factors
        self.base_boxes = torch.zeros(
            [cell_size, cell_size, 1, 4], device=device
        )
        for y in range(cell_size):
            for x in range(cell_size):
                self.base_boxes[y, x, 0, :2] = torch.tensor(
                    [image_size / cell_size * x, image_size / cell_size * y]
                )
        self.base_boxes = self.base_boxes.repeat(1, 1, boxes_per_cell, 1)
        self.predict_boxes_multiple = torch.tensor(
            [
                image_size / cell_size,
                image_size / cell_size,
                image_size,
                image_size,
            ],
            device=device,
        )

    def forward(self, predicts, labels, objects_num):
        loss = 0.0
        batch_size = predicts.shape[0]
        for i in range(batch_size):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            object_num = objects_num[i]

            for j in range(object_num):
                loss += self._calculate_loss(predict, label[j : j + 1, :])

        return loss / batch_size

    def _calculate_loss(self, predict, label):
        """
        calculate loss
        Args:
        predict: 3-D tensor [cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        label : [1, 5]  (x_center, y_center, w, h, class)
        """
        label = label.view(-1)

        # Step A: Calculate object mask
        object_mask = self._calculate_object_mask(label)

        # Step B: Center mask for the object
        center_mask = self._calculate_center_mask(label)

        # Step C: IOU between predicted and true boxes
        predict_boxes = predict[
            :, :, self.num_classes + self.boxes_per_cell :
        ].view(self.cell_size, self.cell_size, self.boxes_per_cell, 4)
        # cell position to pixel position
        predict_boxes = predict_boxes * self.predict_boxes_multiple
        # if there's no predict_box in that cell, then the base_boxes will be calcuated with label and got iou equals 0
        predict_boxes = self.base_boxes + predict_boxes
        iou_predict_truth = self._iou(predict_boxes, label[:4])

        # Calculate tensors C and I
        C = iou_predict_truth * center_mask.unsqueeze(-1)
        I = (
            iou_predict_truth >= iou_predict_truth.max(dim=-1, keepdim=True)[0]
        ).float() * center_mask.unsqueeze(-1)
        no_I = 1 - I

        # Calculate predictions and ground truth
        p_C = predict[
            :, :, self.num_classes : self.num_classes + self.boxes_per_cell
        ]
        # calculate truth x, y, sqrt_w, sqrt_h 0-D
        x, y = label[0], label[1]
        sqrt_w, sqrt_h = torch.sqrt(torch.abs(label[2])), torch.sqrt(
            torch.abs(label[3])
        )
        # calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        p_x, p_y = predict_boxes[..., 0], predict_boxes[..., 1]
        p_sqrt_w, p_sqrt_h = torch.sqrt(
            torch.clamp(predict_boxes[..., 2], min=0, max=self.image_size)
        ), torch.sqrt(
            torch.clamp(predict_boxes[..., 3], min=0, max=self.image_size)
        )
        # calculate ground truth p 1-D tensor [NUM_CLASSES]
        P = F.one_hot(label[4].long(), self.num_classes).float()
        # calculate predicted p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
        p_P = predict[:, :, : self.num_classes]

        # Loss calculations
        class_loss = (
            self._l2_loss(object_mask.unsqueeze(-1) * (p_P - P))
            * self.class_scale
        )
        object_loss = self._l2_loss(I * (p_C - C)) * self.object_scale
        noobject_loss = self._l2_loss(no_I * p_C) * self.noobject_scale
        coord_loss = (
            self._l2_loss(I * (p_x - x) / (self.image_size / self.cell_size))
            + self._l2_loss(I * (p_y - y) / (self.image_size / self.cell_size))
            + self._l2_loss(I * (p_sqrt_w - sqrt_w)) / self.image_size
            + self._l2_loss(I * (p_sqrt_h - sqrt_h)) / self.image_size
        ) * self.coord_scale
        return class_loss + object_loss + noobject_loss + coord_loss

    def _calculate_object_mask(self, label):
        # Converts (x_center, y_center, w, h) to (xmin, ymin, xmax, ymax)
        min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
        max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)
        min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
        max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)
        # Clamp values to [0, cell_size]
        min_x, min_y = torch.floor(min_x), torch.floor(min_y)
        max_x, max_y = torch.ceil(max_x).clamp(max=self.cell_size), torch.ceil(
            max_y
        ).clamp(max=self.cell_size)
        # Create object mask
        object_mask = torch.zeros(
            [self.cell_size, self.cell_size],
            dtype=torch.float32,
            device=self.device,
        )
        object_mask[int(min_y) : int(max_y), int(min_x) : int(max_x)] = 1.0
        return object_mask

    def _calculate_center_mask(self, label):
        center_x = torch.floor(label[0] / (self.image_size / self.cell_size))
        center_y = torch.floor(label[1] / (self.image_size / self.cell_size))
        center_mask = torch.zeros(
            [self.cell_size, self.cell_size],
            dtype=torch.float32,
            device=self.device,
        )
        center_mask[int(center_y), int(center_x)] = 1.0
        return center_mask

    def _iou(self, boxes1, boxes2):
        boxes1 = torch.stack(
            [
                boxes1[..., 0] - boxes1[..., 2] / 2,
                boxes1[..., 1] - boxes1[..., 3] / 2,
                boxes1[..., 0] + boxes1[..., 2] / 2,
                boxes1[..., 1] + boxes1[..., 3] / 2,
            ],
            dim=-1,
        )
        boxes2 = torch.stack(
            [
                boxes2[0] - boxes2[2] / 2,
                boxes2[1] - boxes2[3] / 2,
                boxes2[0] + boxes2[2] / 2,
                boxes2[1] + boxes2[3] / 2,
            ]
        )

        lu = torch.max(boxes1[..., :2], boxes2[:2])
        rd = torch.min(boxes1[..., 2:], boxes2[2:])
        intersection = (rd - lu).clamp(min=0)
        inter_area = intersection[..., 0] * intersection[..., 1]
        square1 = (boxes1[..., 2] - boxes1[..., 0]) * (
            boxes1[..., 3] - boxes1[..., 1]
        )
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
        union_area = square1 + square2 - inter_area + 1e-6

        return inter_area / union_area

    def _l2_loss(self, x):
        return 0.5 * F.mse_loss(x, torch.zeros_like(x), reduction="sum")
