import numpy as np
import torch
from torchvision.ops import nms
epsilon = 1e-8

def xywh_to_xyxy_np(boxes):
    # Convert from [cx, cy, w, h] to [x_min, y_min, x_max, y_max]
    xyxy = np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,  # x - w / 2, y - h / 2
                           boxes[:, :2] + boxes[:, 2:] / 2), 1)  # x + w / 2, y + h / 2
    
    # Clip the coordinates to the range [0, 1]
    return np.clip(xyxy, 0, 1)


def xyxy_to_xywh_np(boxes):
    return np.concatenate(((boxes[:, 2:] + boxes[:, :2]) / 2,  # (min + max) / 2
                            boxes[:, 2:] - boxes[:, :2]), 1)  # max - min

def IoU_np(box_a, box_b):
    min_xy_a, max_xy_a = box_a[:, :2], box_a[:, 2:]
    min_xy_b, max_xy_b = box_b[:, :2], box_b[:, 2:]

    # boardcast [Na, 1, 2] with [1, Nb, 2], result in [Na, Nb, 2]
    # this means compare each box in box_a with each box in box_b
    # thus there are Na x Nb minimum/maximum (x, y)
    max_xy = np.minimum(max_xy_a[:, None, :], max_xy_b[None, :, :])
    min_xy = np.maximum(min_xy_a[:, None, :], min_xy_b[None, :, :])
    # if two boxes do not overlap, the IOU should be 0
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    # [Na, Nb] x [Na, Nb] result in [Na, Nb]
    # which means Na x Nb overlaps for each pair of boxes
    inter = inter[:, :, 0] * inter[:, :, 1]

    # box = [x_min, y_min, x_max, y_max]
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))[:, None]  # [Na, 1]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))[None, :]  # [1, Nb]
    union = area_a + area_b - inter  # [Na, Nb]
    return inter / union  # [Na, Nb]

def match_np(gt_boxes, anchor_boxes, cls_labels):
    # jaccard index
    overlaps = IoU_np(gt_boxes, xywh_to_xyxy_np(anchor_boxes)) # iou value to each ground truth with anchors
    
    best_anchor_idx = np.argmax(overlaps, 1)  # for every ground truth box, find the best matched anchor
    
    # [1, 8732] best ground truth for each anchor
    best_gt_overlap = np.amax(overlaps, 0)  # max iou value to ground truth with anchors
    best_gt_idx = np.argmax(overlaps, 0)  # index of above

    # print(f"clss label = :{cls_labels}")
    # print(f"overlaps = {overlaps}, shape: {overlaps.shape}")
    # print(f"bestanchoridx = {best_anchor_idx}, shape: {best_anchor_idx.shape}")
    # print(f"bestgtidx = {best_gt_idx}, shape: {best_gt_idx.shape}")
    
    # print(f"before: bestgtoverlaps = {best_gt_overlap.tolist()}, shape: {best_gt_overlap.shape}")
    # set the iou of anchors that have matched boxes to 2
    # note that other iou are all below 1
    # best_gt_overlap[best_anchor_idx] = 2

    # ensure every gt matches with its anchor of max overlap
    # best_anchor_idx coresponding to gt box 1, 2, ..., N
    best_gt_idx[best_anchor_idx] = np.arange(best_anchor_idx.shape[0])
    # print(cls_labels.shape)
    # print(best_gt_idx.shape)
    # find the gtboxes of each anchor
    gt_matches = gt_boxes[best_gt_idx]  # Shape: [8732, 4]

    # find the class label of each anchor
    cls_target = cls_labels[best_gt_idx] + 1  # Shape: [8732]
    
    # set label 0 to anchors that have a low iou
    pos_threshold = best_gt_overlap.mean()
    cls_target[best_gt_overlap < pos_threshold] = 0  # label as background
    # print(f"after: bestgtoverlaps = {best_gt_overlap.tolist()}, shape: {best_gt_overlap.shape}")
    # print(f"clstarget = {cls_target.tolist()}, shape: {cls_target.shape}")
    # distance between matched gt box center and anchor's center
    g_cxcy = (gt_matches[:, :2] + gt_matches[:, 2:]) / 2 - anchor_boxes[:, :2]
    # distance / anchor_box_size, and encode variance
    g_cxcy /= (anchor_boxes[:, 2:] * 0.1)
    # matched gt_box_size / anchor_box_size
    g_wh = (gt_matches[:, 2:] - gt_matches[:, :2]) / anchor_boxes[:, 2:]
    # apply log, and encode variance
    g_wh = np.log(np.maximum(g_wh, epsilon)) / 0.2

    # return target for smooth_l1_loss
    reg_target = np.concatenate([g_cxcy, g_wh], 1)  # [8732, 4]

    return reg_target, cls_target



def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=10):
    """
    Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Use only on batched-inputs. Use zero-padding in order to batch output
    results.

    Args:
        scores: Batch x N Tensor/Dictionary containing float scores.
        bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
        nms_threshold: Matching threshold in NMS algorithm;
        keep_top_k: Number of total objects to keep after NMS.
    Returns:
        scores, bboxes Tensors/Dictionaries, sorted by score.
          Padded with zero if necessary.
    """
    def bboxes_nms(scores, bboxes, nms_threshold, keep_top_k):
        # Perform NMS for a single batch
        # print(bboxes)
        keep = torch.ops.torchvision.nms(bboxes, scores, nms_threshold)
        if keep_top_k > 0:
            keep = keep[:keep_top_k]
        return scores[keep], bboxes[keep]

    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        d_scores = {}
        d_bboxes = {}
        for c in scores.keys():
            s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                    nms_threshold=nms_threshold,
                                    keep_top_k=keep_top_k)
            d_scores[c] = s
            d_bboxes[c] = b
        return d_scores, d_bboxes

    # Tensors inputs.
    batch_size = scores.size(0)
    output_scores = []
    output_bboxes = []
    for i in range(batch_size):
        s, b = bboxes_nms(scores[i], bboxes[i], nms_threshold, keep_top_k)
        output_scores.append(s)
        output_bboxes.append(b)

    # Pad the results to keep dimensions consistent
    max_boxes = max(len(s) for s in output_scores)
    padded_scores = torch.zeros(batch_size, max_boxes, device=scores.device)
    padded_bboxes = torch.zeros(batch_size, max_boxes, 4, device=bboxes.device)

    for i in range(batch_size):
        num_boxes = len(output_scores[i])
        padded_scores[i, :num_boxes] = output_scores[i]
        padded_bboxes[i, :num_boxes, :] = output_bboxes[i]

    return padded_scores, padded_bboxes
