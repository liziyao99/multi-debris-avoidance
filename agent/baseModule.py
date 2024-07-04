import torch
from torch import nn

class baseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self._dummy_param.device