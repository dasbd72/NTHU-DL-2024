from typing import Tuple

import torch


def scale(
    x: torch.Tensor,
    range1: Tuple[float, float],
    range2: Tuple[float, float],
    clamp: bool = False,
) -> torch.Tensor:
    """
    Scales the input tensor from range1 to range2

    :param x torch.Tensor: The input tensor
    :param range1 Tuple[float, float]: The range of the input tensor
    :param range2 Tuple[float, float]: The range to scale to
    :param clamp bool: Whether to clamp the output
    """
    x_min, x_max = range1
    y_min, y_max = range2
    y = (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min
    if clamp:
        y = torch.clamp(y, y_min, y_max)
    return x
