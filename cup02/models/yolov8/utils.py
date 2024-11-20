import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion


def make_anchors(
    xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    strides: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grid_cell_offset=0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate anchors from features."""
    dtype, device = xs[0].dtype, xs[0].device  # data type and device
    anchor_points, stride_tensor = [], []  # anchor points and stride tensor
    for i, stride in enumerate(strides):
        _, _, h, w = xs[i].shape
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")  # mesh grid
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def distance2xyxybbox(
    distance: torch.Tensor, anchor_points: torch.Tensor, dim=-1
) -> torch.Tensor:
    """
    Transform distance(ltrb) to box(xyxy). ltrb: left, top, right, bottom.

    :param distance torch.Tensor:
        shape=(batch_size, ..., 4, ...).
        Distance tensor, where at dimension `dim`, the shape is A.
    :param anchor_points torch.Tensor:
        shape=(1, ..., 2, ...).
        Anchor points, where at dimension `dim`, the shape is A.

    :rtype: torch.Tensor
    :return: shape=(batch_size, ..., 4, ...). Bounding boxes.
    """
    lt, rb = distance.chunk(2, dim=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim=dim)


def xyxybbox2distance(
    boxes: torch.Tensor, anchor_points: torch.Tensor, reg_max: int, dim=-1
) -> torch.Tensor:
    """
    Transform box(xyxy) to distance(ltrb).

    :param boxes torch.Tensor:
        shape=(batch_size, ..., 4, ...).
        Bounding boxes, where at dimension `dim`, the shape is A.
    :param anchor_points torch.Tensor:
        shape=(1, ..., 2, ...).
        Anchor points, where at dimension `dim`, the shape is A.

    :rtype: torch.Tensor
    :return: shape=(batch_size, ..., 4, ...). Distance tensor.
    """
    x1y1, x2y2 = boxes.chunk(2, dim=dim)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    distance = torch.cat((lt, rb), dim=dim).clamp(
        min=0, max=reg_max - 1 - 0.01
    )
    return distance


def compute_ciou(
    box_preds: torch.Tensor, box_labels: torch.Tensor, eps=1e-7
) -> torch.Tensor:
    """
    Compute the Complete Intersection over Union (CIoU) of two set of boxes.

    $$
    CIoU = IoU - (center_distance / enclose_diagonal) - aspect_ratio_consistency
    $$

    :param box_preds torch.Tensor: shape=(batch_size, A, 4). Predicted bounding boxes.
    :param box_labels torch.Tensor: shape=(batch_size, A, 4,). Target bounding boxes.

    :rtype: torch.Tensor
    :return: shape=(batch_size, N). CIoU values.
    """
    assert (
        box_preds.device == box_labels.device
    ), "The devices of two tensors must be the same."
    assert (
        box_preds.shape == box_labels.shape
    ), "The shapes of two tensors must be the same."

    b1_x1, b1_y1, b1_x2, b1_y2 = torch.chunk(box_preds, 4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = torch.chunk(box_labels, 4, dim=-1)
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection Area
    inner_left, inner_right = torch.max(b1_x1, b2_x1), torch.min(b1_x2, b2_x2)
    inner_up, inner_down = torch.max(b1_y1, b2_y1), torch.min(b1_y2, b2_y2)
    intersection = torch.clamp(inner_right - inner_left, min=0) * torch.clamp(
        inner_down - inner_up, min=0
    )
    # Union Area = A + B - Intersection
    union = b1_w * b1_h + b2_w * b2_h - intersection
    # IoU = Intersection / Union
    iou = intersection / (union + eps)

    # Calculate the center distance
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    center_distance = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Calculate diagonal length of the smallest enclosing box
    enclose_left, enclose_right = torch.min(b1_x1, b2_x1), torch.max(
        b1_x2, b2_x2
    )
    enclose_up, enclose_down = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
    enclose_w, enclose_h = (
        enclose_right - enclose_left,
        enclose_down - enclose_up,
    )
    enclose_diagonal = enclose_w**2 + enclose_h**2

    # Calculate the aspect ratio consistency term
    v = (4 / (torch.pi**2)) * (
        torch.atan(b2_w / (b2_h + eps)) - torch.atan(b1_w / (b1_h + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)

    # Calculate CIoU
    ciou = iou - (center_distance / (enclose_diagonal + eps)) - alpha * v
    ciou = ciou.squeeze(-1)  # shape=(batch_size, N)
    return ciou


def scale_xyxy(boxes: torch.Tensor, scale: float, dim=-1) -> torch.Tensor:
    """
    Scale the bounding boxes.

    :param boxes torch.Tensor:
        shape=(batch_size, ..., 4, ...).
        Bounding boxes, where at dimension `dim`, the shape is A.
    :param scale float:
        The scale factor.
    :param dim int:
        The dimension to split the bounding boxes.

    :rtype: torch.Tensor
    :return: shape=(batch_size, ..., 4, ...). Scaled bounding boxes.
    """
    x1y1, x2y2 = boxes.chunk(2, dim=dim)
    center = (x1y1 + x2y2) / 2 * scale
    size = (x2y2 - x1y1) * scale
    x1y1 = center - size / 2
    x2y2 = center + size / 2
    return torch.cat((x1y1, x2y2), dim=dim)


def process_outputs(
    outputs,
    image_size,
    conf_threshold=0.0,
    conf_ratio=0.5,
    per_class_conf_ratio: list[float] = None,
    fuse_boxes=True,
    iou_thr=0.5,
    skip_box_thr=0.0,
):
    """
    Process YOLO outputs into bounding boxes, class, and confidence, applying weighted boxes fusion.

    :param outputs torch.Tensor: YOLO inference outputs
        shape=(batch_size, num_anchors, 6), where the last dimension is
        (x1, y1, x2, y2, class, confidence)
    :param image_size int: The image size.
    :param conf_threshold float: Confidence threshold for filtering boxes.
    :param conf_ratio float: Confidence ratio for filtering boxes, relative to max confidence.
    :param per_class_conf_ratio list[float]: Per class confidence ratio for filtering boxes.
    :param iou_thr float: IOU threshold for WBF.
    :param skip_box_thr float: Minimum confidence threshold for boxes to consider in WBF.
    """
    # Split outputs into bbox, class, and confidence, then flatten batch and anchors
    bbox_preds, class_preds, conf_preds = outputs.split(
        [4, 1, 1], dim=-1
    )  # (batch_size, A, 4), (batch_size, A, 1), (batch_size, A, 1)
    batch_size, num_anchors, _ = outputs.shape

    # Get max confidence of each batch
    max_conf = conf_preds.max(dim=1, keepdim=True)[0].repeat(
        1, num_anchors, 1
    )  # (batch_size, A, 1)

    # Flatten for batch processing
    bbox_preds = bbox_preds.view(-1, 4)
    class_preds = class_preds.view(-1)
    conf_preds = conf_preds.view(-1)
    max_conf = max_conf.view(-1)

    # Filter based on confidence threshold
    mask = conf_preds > conf_threshold
    if per_class_conf_ratio is not None:
        for i, cls_conf_ratio in enumerate(per_class_conf_ratio):
            mask = mask & (
                (class_preds != i)
                | (
                    (class_preds == i)
                    & (conf_preds > (max_conf * cls_conf_ratio))
                )
            )
    else:
        mask = mask & (conf_preds > (max_conf * conf_ratio))
    filtered_bboxes = bbox_preds[mask]
    filtered_classes = class_preds[mask].int()
    filtered_confs = conf_preds[mask]

    # Normalize bounding boxes to [0, 1] for WBF
    normalized_bboxes = filtered_bboxes / image_size

    # Split results back by batch
    batch_splits = mask.view(batch_size, num_anchors).sum(dim=1).tolist()
    bbox_list = torch.split(normalized_bboxes, batch_splits)
    class_list = torch.split(filtered_classes, batch_splits)
    conf_list = torch.split(filtered_confs, batch_splits)

    # Convert tensors to lists for WBF compatibility
    bbox_list = [b.tolist() for b in bbox_list]
    class_list = [c.tolist() for c in class_list]
    conf_list = [conf.tolist() for conf in conf_list]
    if not fuse_boxes:
        return bbox_list, class_list, conf_list

    fused_boxes, fused_classes, fused_scores = [], [], []

    # Apply WBF for each batch element
    for bboxes, classes, confs in zip(bbox_list, class_list, conf_list):
        boxes, scores, labels = weighted_boxes_fusion(
            [bboxes],
            [confs],
            [classes],
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        # Denormalize bounding boxes back to original image size
        boxes = (torch.tensor(boxes) * image_size).tolist()
        fused_boxes.append(boxes)
        fused_classes.append(labels.astype(np.long))
        fused_scores.append(scores)

    return fused_boxes, fused_classes, fused_scores
