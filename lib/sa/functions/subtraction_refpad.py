import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from lib.sa.functions.utils import Dtype, Stream, load_kernel


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_subtraction_refpad_forward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction_refpad_forward_kernel(
const ${Dtype}* bottom_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${input_channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${input_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int h_in_center = -${pad_h} + h * ${stride_h} + (${kernel_h} - 1) / 2 * ${dilation_h};
    const int w_in_center = -${pad_w} + w * ${stride_w} + (${kernel_w} - 1) / 2 * ${dilation_w};
    const int offset_center = ((n * ${input_channels} + c) * ${bottom_height} + h_in_center) * ${bottom_width} + w_in_center;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
        int offset_bottom;
        if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
          offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
        }
        else {
          if (h_in < 0) h_in = -h_in;
          if (h_in >= ${bottom_height}) h_in = 2 * (${bottom_height} - 1) - h_in;
          if (w_in < 0) w_in = -w_in;
          if (w_in >= ${bottom_width}) w_in = 2 * (${bottom_width} - 1) - w_in;
          offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
        }
        top_data[offset_top] = bottom_data[offset_center] - bottom_data[offset_bottom];
      }
    }
  }
}
'''


_subtraction_refpad_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void subtraction_refpad_input_backward_kernel(
    const ${Dtype}* const top_diff, ${Dtype}* bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${input_channels} / (${bottom_height} + 2 * ${pad_h}) / (${bottom_width} + 2 * ${pad_w});
    const int c = (index / (${bottom_height} + 2 * ${pad_h}) / (${bottom_width} + 2 * ${pad_w})) % ${input_channels};
    const int h = (index / (${bottom_width} + 2 * ${pad_w})) % (${bottom_height} + 2 * ${pad_h});
    const int w = index % (${bottom_width} + 2 * ${pad_w});
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h - kh * ${dilation_h};
        const int w_out_s = w - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
            value += -top_diff[offset_top];
          }
        }
      }
    }
    const int hh = h - ${pad_h};
    const int ww = w - ${pad_w};
    if ((hh >= 0) && (hh < ${bottom_height}) && (ww >= 0) && (ww < ${bottom_width})) {
      if (((hh % ${stride_h}) == 0) && ((ww % ${stride_w}) == 0)) {
        const int h_out = hh / ${stride_h};
        const int w_out = ww / ${stride_w};
        for (int kh = 0; kh < ${kernel_h}; ++kh) {
          for (int kw = 0; kw < ${kernel_w}; ++kw) {
            const int offset_top = ((n * ${input_channels} + c) * ${kernel_h} * ${kernel_w} + (kh * ${kernel_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
            value += top_diff[offset_top];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''


class SubtractionRefpad(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation = kernel_size, stride, padding, dilation
        assert input.dim() == 4 and input.is_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        output = input.new(batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
        n = output.numel() // output.shape[2]
        with torch.cuda.device_of(input):
            f = load_kernel('subtraction_refpad_forward_kernel', _subtraction_refpad_forward_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, input_channels=input_channels,
                            bottom_height=input_height, bottom_width=input_width,
                            top_height=output_height, top_width=output_width,
                            kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel_size, stride, padding, dilation = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation
        input, = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        batch_size, input_channels, input_height, input_width = input.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        grad_input = None
        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, input_channels=input_channels,
                   bottom_height=input_height, bottom_width=input_width,
                   top_height=output_height, top_width=output_width,
                   kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(batch_size, input_channels, input_height + 2 * padding[0], input_width + 2 * padding[1])
                n = grad_input.numel()
                opt['nthreads'] = n
                f = load_kernel('subtraction_refpad_input_backward_kernel', _subtraction_refpad_input_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                grad_input[:, :, padding[0] + 1:2 * padding[0] + 1, :] += torch.flip(grad_input[:, :, :padding[0], :], dims=[2])
                grad_input[:, :, input_width - 1:input_width + padding[0] - 1, :] += torch.flip(grad_input[:, :, input_width + padding[0]:, :], dims=[2])
                grad_input[:, :, :, padding[1] + 1:2 * padding[1] + 1] += torch.flip(grad_input[:, :, :, :padding[1]], dims=[3])
                grad_input[:, :, :, input_width - 1:input_width + padding[1] - 1] += torch.flip(grad_input[:, :, :, input_width + padding[1]:], dims=[3])
                grad_input = grad_input[:, :, padding[0]:padding[0] + input_width, padding[1]:padding[1] + input_width]
        return grad_input, None, None, None, None


def subtraction_refpad(input, kernel_size=3, stride=1, padding=0, dilation=1):
    assert input.dim() == 4
    if input.is_cuda:
        out = SubtractionRefpad.apply(input, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out


def test_subtraction_refpad():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    kernel_size, stride, dilation = 5, 4, 2
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c, in_height, in_width = 2, 8, 5, 5
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    x = torch.randn(n, c, in_height, in_width, requires_grad=True).double().cuda()

    y1 = subtraction_refpad(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    unfold_i = torch.nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
    unfold_j = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
    pad = torch.nn.ReflectionPad2d(padding)
    y2 = unfold_i(x).view(n, c, 1, out_height * out_width) - unfold_j(pad(x)).view(n, c, pow(kernel_size, 2), out_height * out_width)
    assert (y1 - y2).abs().max() < 1e-9

    gx1 = torch.autograd.grad(y1.mean(), x, retain_graph=True)[0]
    gx2 = torch.autograd.grad(y2.mean(), x, retain_graph=True)[0]
    assert (gx1 - gx2).abs().max() < 1e-9

    from functools import partial
    assert torch.autograd.gradcheck(partial(subtraction_refpad, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation), x)
    print('test case passed')


if __name__ == '__main__':
    test_subtraction_refpad()
