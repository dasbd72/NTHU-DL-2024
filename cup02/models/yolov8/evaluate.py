import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from compute_score import compute_score
from models.yolov8.data import TestDatasetGenerator
from models.yolov8.layers import YoloV8
from models.yolov8.utils import process_outputs
from utils.evaluate import evaluate


def predict_all(
    model: YoloV8,
    test_data_path: str,
    test_image_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = False,
    device: torch.device = torch.device("cuda"),
):
    """
    Predict the bounding boxes on the test data and save the predictions to a file.

    :param model YoloV8: The YOLOv8 model.
    :param test_data_path str: The path to the test data.
    :param test_image_dir str: The directory containing the test images.
    :param image_size int: The image size.
    :param batch_size int: The batch size.
    :param num_workers int: The number of workers for the data loader.
    :param pin_memory bool: Whether to pin memory for the data loader.
    :param device torch.device: The device to use.

    :rtype: tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]
    :return: image_names, image_heights, image_widths, outputs
    """
    # Test data loader
    data_loader = DataLoader(
        TestDatasetGenerator(test_data_path, test_image_dir, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    all_image_names = []
    all_image_heights = torch.tensor([])
    all_image_widths = torch.tensor([])
    all_outputs = torch.tensor([])
    for image_names, images, image_heights, image_widths in tqdm(data_loader):
        all_image_names.extend(image_names)
        all_image_heights = torch.cat(
            (all_image_heights, image_heights), dim=0
        )
        all_image_widths = torch.cat((all_image_widths, image_widths), dim=0)

        images, image_heights, image_widths = (
            images.to(device),
            image_heights.to(device),
            image_widths.to(device),
        )
        with torch.no_grad():
            model.eval()
            outputs = model.inference(images).cpu().detach()
        all_outputs = torch.cat((all_outputs, outputs), dim=0)
    return all_image_names, all_image_heights, all_image_widths, all_outputs


def evaluate_all(
    image_names: list[str],
    image_heights: torch.Tensor,
    image_widths: torch.Tensor,
    outputs: torch.Tensor,
    image_size: int,
    pred_output_path: str,
    eval_output_path: str,
    conf_threshold: float = 0.0,
    conf_ratio: float = 0.1,
    per_class_conf_ratio: list[float] = None,
    iou_thr: float = 0.5,
):
    """
    Evaluate the predictions and save the results to a file.

    :param image_names list[str]: The image names.
    :param image_heights torch.Tensor: The image heights.
    :param image_widths torch.Tensor: The image widths.
    :param outputs torch.Tensor: The model outputs.
    :param image_size int: The image size.
    :param pred_output_path str: The path to save the predictions.
    :param eval_output_path str: The path to save the evaluation results.
    :param conf_threshold float: The confidence threshold.
    :param conf_ratio float: The confidence ratio.
    :param per_class_conf_ratio list[float]: The per class confidence ratio.
    :param iou_thr float: The IOU threshold.

    :rtype: float
    :return: The evaluation score.
    """
    bbox_list, class_list, conf_list = process_outputs(
        outputs,
        image_size,
        conf_threshold=conf_threshold,
        conf_ratio=conf_ratio,
        per_class_conf_ratio=per_class_conf_ratio,
        iou_thr=iou_thr,
    )
    if not os.path.exists(os.path.dirname(pred_output_path)):
        os.makedirs(os.path.dirname(pred_output_path))
    with open(pred_output_path, "w") as output_file:
        for image_idx in range(len(image_names)):
            answers = [
                image_names[image_idx],
            ]
            bboxes, classes, confidences = (
                bbox_list[image_idx],
                class_list[image_idx],
                conf_list[image_idx],
            )
            for bbox, class_idx, conf in zip(bboxes, classes, confidences):
                x1, y1, x2, y2 = bbox
                # Convert normalized bounding box to original image size
                x1, y1, x2, y2 = (
                    x1 * (image_widths[image_idx] / image_size),
                    y1 * (image_heights[image_idx] / image_size),
                    x2 * (image_widths[image_idx] / image_size),
                    y2 * (image_heights[image_idx] / image_size),
                )
                # Clip bounding box coordinates to image size
                x1 = max(0, min(image_widths[image_idx] - 1, x1))
                y1 = max(0, min(image_heights[image_idx] - 1, y1))
                x2 = max(0, min(image_widths[image_idx], x2))
                y2 = max(0, min(image_heights[image_idx], y2))
                answers.append(
                    "%d %d %d %d %d %f" % (x1, y1, x2, y2, class_idx, conf)
                )
            output_file.write(" ".join(answers) + "\n")
    evaluate(pred_output_path, eval_output_path)
    score = compute_score(eval_output_path)
    return score


def predict_and_evaluate(
    model: YoloV8,
    test_data_path: str,
    test_image_dir: str,
    image_size: int,
    batch_size: int,
    pred_output_path: str,
    eval_output_path: str,
    num_workers: int = 8,
    pin_memory: bool = False,
    conf_threshold: float = 0.0,
    conf_ratio: float = 0.1,
    per_class_conf_ratio: list[float] = None,
    iou_thr: float = 0.5,
    device: torch.device = torch.device("cuda"),
):
    """
    Predict the bounding boxes on the test data and save the predictions to a file.
    """
    image_names, image_heights, image_widths, outputs = predict_all(
        model,
        test_data_path,
        test_image_dir,
        image_size,
        batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
    )
    score = evaluate_all(
        image_names,
        image_heights,
        image_widths,
        outputs,
        image_size,
        pred_output_path,
        eval_output_path,
        conf_threshold=conf_threshold,
        conf_ratio=conf_ratio,
        per_class_conf_ratio=per_class_conf_ratio,
        iou_thr=iou_thr,
    )
    return score
