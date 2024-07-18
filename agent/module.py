import torch
from torch import nn

class sortReLU(nn.ReLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim()<3:
            raise NotImplementedError("sortReLU only support 3D input")
        input = input.sort(dim=-2, descending=True)[0]
        return super().forward(input)
    
class softmaxReLU(nn.ReLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim()<3:
            raise NotImplementedError("softmaxReLU only support 3D input")
        input = input*torch.softmax(input, dim=-2)
        return super().forward(input)
    
class layerNormReLU(nn.ReLU):
    def __init__(self, inplace: bool = False, eps=1e-6):
        super().__init__(inplace)
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim()<3:
            raise NotImplementedError("layerNormReLU only support 3D input")
        mean = input.mean(dim=-2, keepdim=True)
        var = input.var(dim=-2, keepdim=True) + self.eps
        input = (input-mean)/torch.sqrt(var)
        return super().forward(input)