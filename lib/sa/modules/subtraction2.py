from torch import nn
from torch.nn.modules.utils import _pair

from .. import functional as F


class Subtraction2(nn.Module):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction2, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def forward(self, input1, input2):
        return F.subtraction2(input1, input2, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
