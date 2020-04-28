from torch import nn
from torch.nn.modules.utils import _pair

from .. import functional as F


class Subtraction(nn.Module):

    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def forward(self, input):
        return F.subtraction(input, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)
