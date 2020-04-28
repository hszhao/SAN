from . import functions


def aggregation(input, weight, kernel_size=3, stride=1, padding=0, dilation=1, pad_mode=1):
    assert input.shape[0] == weight.shape[0] and (input.shape[1] % weight.shape[1] == 0) and pad_mode in [0, 1]
    if input.is_cuda:
        if pad_mode == 0:
            out = functions.aggregation_zeropad(input, weight, kernel_size, stride, padding, dilation)
        elif pad_mode == 1:
            out = functions.aggregation_refpad(input, weight, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out


def subtraction(input, kernel_size=3, stride=1, padding=0, dilation=1, pad_mode=1):
    assert input.dim() == 4 and pad_mode in [0, 1]
    if input.is_cuda:
        if pad_mode == 0:
            out = functions.subtraction_zeropad(input, kernel_size, stride, padding, dilation)
        elif pad_mode == 1:
            out = functions.subtraction_refpad(input, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out


def subtraction2(input1, input2, kernel_size=3, stride=1, padding=0, dilation=1, pad_mode=1):
    assert input1.dim() == 4 and input2.dim() == 4 and pad_mode in [0, 1]
    if input1.is_cuda:
        if pad_mode == 0:
            out = functions.subtraction2_zeropad(input1, input2, kernel_size, stride, padding, dilation)
        elif pad_mode == 1:
            out = functions.subtraction2_refpad(input1, input2, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out
