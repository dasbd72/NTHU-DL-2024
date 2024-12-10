import torch
import torch.nn as nn


def uniform_init(tensor: torch.Tensor, scale: float = 1.0):
    # Calculate the fan-in (number of input units)
    fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")

    # Calculate the variance scale
    bound = torch.sqrt(torch.tensor(3.0) * scale / fan_in)

    # Initialize with uniform distribution
    nn.init.uniform_(tensor, -bound, bound)


class UniformInitConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        init_scale=1,
    ):
        super(UniformInitConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )

        # Initialize the weights
        uniform_init(self.conv.weight, scale=init_scale)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, in_channels, h, w)
        :return torch.Tensor: (batch_size, out_channels, h, w)
        """
        x = self.conv(x)  # (batch_size, out_channels, h, w)
        return x


class UniformInitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_scale=1):
        super(UniformInitLinear, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Initialize the weights
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor):
        """
        :param x torch.Tensor: (batch_size, in_features)
        :return torch.Tensor: (batch_size, out_features)
        """
        x = self.linear(x)  # (batch_size, out_features)
        return x
