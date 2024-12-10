from typing import Dict

import torch.nn as nn

StateDict = Dict[str, nn.Parameter]


class EMAStateDict:
    def __init__(self, state_dict=None, decay=0.9999, device="cpu"):
        self.decay = decay
        self.device = device
        self.shadow: StateDict = {}

        if state_dict is not None:
            self.load_state_dict(state_dict)

    def update(self, state_dict: StateDict):
        for k, v in state_dict.items():
            self.shadow[k].data.mul_(self.decay).add_(
                v.data, alpha=1.0 - self.decay
            )

    def state_dict(self) -> StateDict:
        return self.shadow

    def load_state_dict(self, state_dict: StateDict):
        for k, v in state_dict.items():
            self.shadow[k] = v.clone().detach().to(self.device)
