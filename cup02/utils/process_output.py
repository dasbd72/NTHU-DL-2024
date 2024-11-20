import torch
from ensemble_boxes import *
# Merge boxes for single model predictions

def process_outputs(outputs, CELL_SIZE, NUM_CLASSES, BOXES_PER_CELL, IMAGE_SIZE, conf_threshold=0.0, iou_thr=0.2, method="WBF", weight=None):
    """
    Process YOLO outputs into bounding boxes, class, and confidence
    """
    class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
    conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL

    # Reshape and split outputs
    class_probs = outputs[:, :class_end].view(
        -1, CELL_SIZE, CELL_SIZE, NUM_CLASSES
    )
    confs = outputs[:, class_end:conf_end].view(
        -1, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL
    )
    boxes = outputs[:, conf_end:].view(
        -1, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL * 4
    )
    predicts = torch.cat([class_probs, confs, boxes], dim=3)

    # Extract components for the first image in batch
    p_classes = predicts[0, :, :, :NUM_CLASSES]
    C = predicts[0, :, :, NUM_CLASSES : NUM_CLASSES + BOXES_PER_CELL]
    coordinates = predicts[0, :, :, NUM_CLASSES + BOXES_PER_CELL :]

    # Reshape for element-wise multiplication to calculate P
    p_classes = p_classes.view(CELL_SIZE, CELL_SIZE, 1, NUM_CLASSES)
    C = C.view(CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 1)
    P = (
        C * p_classes
    )  # Shape [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, NUM_CLASSES]

    # Initialize lists to store results
    bboxes = []
    classes = []
    confidences = []

    # Iterate through each box and apply the confidence threshold
    for i in range(CELL_SIZE):
        for j in range(CELL_SIZE):
            for b in range(BOXES_PER_CELL):
                max_conf, class_idx = torch.max(P[i, j, b], dim=-1)

                if max_conf <= conf_threshold:
                    continue

                # Reshape coordinates for easier access
                coordinates = coordinates.view(
                    CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4
                )
                bbox = coordinates[i, j, b, :]

                # Calculate bounding box dimensions
                xcenter = bbox[0]
                ycenter = bbox[1]
                w = bbox[2]
                h = bbox[3]

                xcenter = (j + xcenter.item()) * (
                    IMAGE_SIZE / float(CELL_SIZE)
                )
                ycenter = (i + ycenter.item()) * (
                    IMAGE_SIZE / float(CELL_SIZE)
                )
                w = w.item() * IMAGE_SIZE
                h = h.item() * IMAGE_SIZE

                xmin = xcenter - w / 2.0
                ymin = ycenter - h / 2.0
                xmax = xmin + w
                ymax = ymin + h

                # Append results to lists
                bboxes.append((xmin, ymin, xmax, ymax))
                classes.append(class_idx.item())
                confidences.append(max_conf.item())

    # Sort results by confidence
    sorted_indices = torch.argsort(
        torch.tensor(confidences), descending=True
    ).tolist()
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        normalized_bbox = [
            x1 / IMAGE_SIZE, 
            y1 / IMAGE_SIZE, 
            x2 / IMAGE_SIZE, 
            y2 / IMAGE_SIZE
        ]
        normalized_bboxes.append(normalized_bbox)

    # Wrap in lists to prepare for `weighted_boxes_fusion`
    normalized_bboxes = [normalized_bboxes]
    classes = [classes[i] for i in sorted_indices]
    confidences = [confidences[i] for i in sorted_indices]
    bboxes = [bboxes]  # Wraps bboxes in a list
    confidences = [confidences]  # Wraps confidences in a list
    classes = [classes]  # Wraps classes in a list
    if method == "WBF":
        boxes, scores, labels = weighted_boxes_fusion(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "NMS":
        boxes, scores, labels = nms(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "SoftNMS":
        boxes, scores, labels = soft_nms(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "NMW":
        boxes, scores, labels = non_maximum_weighted(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    else :
        print("Error: Unknow method")
    denormalized_boxes = [
        [
            int(box[0] * IMAGE_SIZE),  # x1
            int(box[1] * IMAGE_SIZE), # y1
            int(box[2] * IMAGE_SIZE),  # x2
            int(box[3] * IMAGE_SIZE)  # y2
        ]
        for box in boxes
    ]
    return denormalized_boxes, labels, scores

def process_outputsV2(outputs, CELL_SIZE, NUM_CLASSES, BOXES_PER_CELL, IMAGE_SIZE, conf_ratio = 0.5, conf_lb = 0.05, iou_thr=0.2, method="WBF", weight=None):
    """
    Process YOLO outputs into bounding boxes, class, and confidence, dynamicly adjust the conf threshold
    """
    class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
    conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL

    # Reshape and split outputs
    class_probs = outputs[:, :class_end].view(
        -1, CELL_SIZE, CELL_SIZE, NUM_CLASSES
    )
    confs = outputs[:, class_end:conf_end].view(
        -1, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL
    )
    boxes = outputs[:, conf_end:].view(
        -1, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL * 4
    )
    predicts = torch.cat([class_probs, confs, boxes], dim=3)

    # Extract components for the first image in batch
    p_classes = predicts[0, :, :, :NUM_CLASSES]
    C = predicts[0, :, :, NUM_CLASSES : NUM_CLASSES + BOXES_PER_CELL]
    coordinates = predicts[0, :, :, NUM_CLASSES + BOXES_PER_CELL :]

    # Reshape for element-wise multiplication to calculate P
    p_classes = p_classes.view(CELL_SIZE, CELL_SIZE, 1, NUM_CLASSES)
    C = C.view(CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 1)
    P = (
        C * p_classes
    )  # Shape [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, NUM_CLASSES]

    # Initialize lists to store results
    bboxes = []
    classes = []
    confidences = []

    # Iterate through each box and apply the confidence threshold
    max_conf_picture = 0
    for i in range(CELL_SIZE):
        for j in range(CELL_SIZE):
            for b in range(BOXES_PER_CELL):
                max_conf, class_idx = torch.max(P[i, j, b], dim=-1)
                max_conf_picture = max(max_conf, max_conf_picture)
    
    for i in range(CELL_SIZE):
        for j in range(CELL_SIZE):
            for b in range(BOXES_PER_CELL):
                max_conf, class_idx = torch.max(P[i, j, b], dim=-1)

                if max_conf <= max_conf_picture * conf_ratio:
                    continue
                if max_conf < conf_lb:
                    continue
                # Reshape coordinates for easier access
                coordinates = coordinates.view(
                    CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4
                )
                bbox = coordinates[i, j, b, :]

                # Calculate bounding box dimensions
                xcenter = bbox[0]
                ycenter = bbox[1]
                w = bbox[2]
                h = bbox[3]

                xcenter = (j + xcenter.item()) * (
                    IMAGE_SIZE / float(CELL_SIZE)
                )
                ycenter = (i + ycenter.item()) * (
                    IMAGE_SIZE / float(CELL_SIZE)
                )
                w = w.item() * IMAGE_SIZE
                h = h.item() * IMAGE_SIZE

                xmin = xcenter - w / 2.0
                ymin = ycenter - h / 2.0
                xmax = xmin + w
                ymax = ymin + h

                # Append results to lists
                bboxes.append((xmin, ymin, xmax, ymax))
                classes.append(class_idx.item())
                confidences.append(max_conf.item())

    # Sort results by confidence
    sorted_indices = torch.argsort(
        torch.tensor(confidences), descending=True
    ).tolist()
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        normalized_bbox = [
            x1 / IMAGE_SIZE, 
            y1 / IMAGE_SIZE, 
            x2 / IMAGE_SIZE, 
            y2 / IMAGE_SIZE
        ]
        normalized_bboxes.append(normalized_bbox)

    # Wrap in lists to prepare for `weighted_boxes_fusion`
    normalized_bboxes = [normalized_bboxes]
    classes = [classes[i] for i in sorted_indices]
    confidences = [confidences[i] for i in sorted_indices]
    bboxes = [bboxes]  # Wraps bboxes in a list
    confidences = [confidences]  # Wraps confidences in a list
    classes = [classes]  # Wraps classes in a list
    if method == "WBF":
        boxes, scores, labels = weighted_boxes_fusion(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "NMS":
        boxes, scores, labels = nms(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "SoftNMS":
        boxes, scores, labels = soft_nms(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "NMW":
        boxes, scores, labels = non_maximum_weighted(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    else :
        print("Error: Unknow method")
    denormalized_boxes = [
        [
            int(box[0] * IMAGE_SIZE),  # x1
            int(box[1] * IMAGE_SIZE), # y1
            int(box[2] * IMAGE_SIZE),  # x2
            int(box[3] * IMAGE_SIZE)  # y2
        ]
        for box in boxes
    ]
    return denormalized_boxes, labels, scores

def process_outputsV3(outputs, CELL_SIZE, NUM_CLASSES, BOXES_PER_CELL, IMAGE_SIZE, conf_list = [], iou_thr=0.2, method="WBF", weight=None):
    """
    Process YOLO outputs into bounding boxes, class, and confidence
    """
    class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
    conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL

    # Reshape and split outputs
    class_probs = outputs[:, :class_end].view(
        -1, CELL_SIZE, CELL_SIZE, NUM_CLASSES
    )
    confs = outputs[:, class_end:conf_end].view(
        -1, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL
    )
    boxes = outputs[:, conf_end:].view(
        -1, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL * 4
    )
    predicts = torch.cat([class_probs, confs, boxes], dim=3)

    # Extract components for the first image in batch
    p_classes = predicts[0, :, :, :NUM_CLASSES]
    C = predicts[0, :, :, NUM_CLASSES : NUM_CLASSES + BOXES_PER_CELL]
    coordinates = predicts[0, :, :, NUM_CLASSES + BOXES_PER_CELL :]

    # Reshape for element-wise multiplication to calculate P
    p_classes = p_classes.view(CELL_SIZE, CELL_SIZE, 1, NUM_CLASSES)
    C = C.view(CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 1)
    P = (
        C * p_classes
    )  # Shape [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, NUM_CLASSES]

    # Initialize lists to store results
    bboxes = []
    classes = []
    confidences = []
    conf_list = [0.05] * 20 # default all 0.05

    # Iterate through each box and apply the confidence threshold
    for i in range(CELL_SIZE):
        for j in range(CELL_SIZE):
            for b in range(BOXES_PER_CELL):
                max_conf, class_idx = torch.max(P[i, j, b], dim=-1)

                if max_conf <= conf_list[class_idx]:
                    continue

                # Reshape coordinates for easier access
                coordinates = coordinates.view(
                    CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4
                )
                bbox = coordinates[i, j, b, :]

                # Calculate bounding box dimensions
                xcenter = bbox[0]
                ycenter = bbox[1]
                w = bbox[2]
                h = bbox[3]

                xcenter = (j + xcenter.item()) * (
                    IMAGE_SIZE / float(CELL_SIZE)
                )
                ycenter = (i + ycenter.item()) * (
                    IMAGE_SIZE / float(CELL_SIZE)
                )
                w = w.item() * IMAGE_SIZE
                h = h.item() * IMAGE_SIZE

                xmin = xcenter - w / 2.0
                ymin = ycenter - h / 2.0
                xmax = xmin + w
                ymax = ymin + h

                # Append results to lists
                bboxes.append((xmin, ymin, xmax, ymax))
                classes.append(class_idx.item())
                confidences.append(max_conf.item())

    # Sort results by confidence
    sorted_indices = torch.argsort(
        torch.tensor(confidences), descending=True
    ).tolist()
    normalized_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        normalized_bbox = [
            x1 / IMAGE_SIZE, 
            y1 / IMAGE_SIZE, 
            x2 / IMAGE_SIZE, 
            y2 / IMAGE_SIZE
        ]
        normalized_bboxes.append(normalized_bbox)

    # Wrap in lists to prepare for `weighted_boxes_fusion`
    normalized_bboxes = [normalized_bboxes]
    classes = [classes[i] for i in sorted_indices]
    confidences = [confidences[i] for i in sorted_indices]
    bboxes = [bboxes]  # Wraps bboxes in a list
    confidences = [confidences]  # Wraps confidences in a list
    classes = [classes]  # Wraps classes in a list
    if method == "WBF":
        boxes, scores, labels = weighted_boxes_fusion(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "NMS":
        boxes, scores, labels = nms(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "SoftNMS":
        boxes, scores, labels = soft_nms(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    elif method == "NMW":
        boxes, scores, labels = non_maximum_weighted(normalized_bboxes, confidences, classes, weights=weight, iou_thr=iou_thr)
    else :
        print("Error: Unknow method")
    denormalized_boxes = [
        [
            int(box[0] * IMAGE_SIZE),  # x1
            int(box[1] * IMAGE_SIZE), # y1
            int(box[2] * IMAGE_SIZE),  # x2
            int(box[3] * IMAGE_SIZE)  # y2
        ]
        for box in boxes
    ]
    return denormalized_boxes, labels, scores