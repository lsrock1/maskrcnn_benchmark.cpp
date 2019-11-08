#include <conv2d.h>

namespace rcnn {
namespace layers {

torch::Tensor Conv2dImpl::forward(const torch::Tensor& input) {
  if (input.numel() > 0) {
    return torch::nn::Conv2dImpl::forward(input);
  }

  int64_t stride = static_cast<torch::ArrayRef<int64_t>>(options.stride()).at(0),
          padding = static_cast<torch::ArrayRef<int64_t>>(options.padding()).at(0),
          dilation = static_cast<torch::ArrayRef<int64_t>>(options.dilation()).at(0),
          output_padding = static_cast<torch::ArrayRef<int64_t>>(options.output_padding()).at(0),
          kernel_size = static_cast<torch::ArrayRef<int64_t>>(options.kernel_size()).at(0);

  std::vector<int64_t> shape;

  if (options.transposed()) {
    int64_t height = (input.size(2) - 1) * stride - 2 * padding + (dilation * (kernel_size - 1) + 1) + output_padding;
    int64_t width = (input.size(3) - 1) * stride - 2 * padding + (dilation * (kernel_size - 1) + 1) + output_padding;
    shape = {input.size(0), bias.size(0), height, width};
  }
  else {
    int64_t height = (input.size(2) + 2 *  padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    int64_t width = (input.size(3) + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    shape = {input.size(0), weight.size(0), height, width};
  }
  return _NewEmptyTensorOp::apply(input, shape);
};

torch::Tensor interpolate(torch::Tensor input, torch::IntArrayRef size/*, float scale_factor, std::string mode, bool align_corners*/) {
  if (input.numel() > 0) {
    return torch::upsample_nearest2d(input, size);
  }
  else {
    return _NewEmptyTensorOp::apply(input, std::vector<int64_t>{input.size(0), input.size(1), size[0], size[1]});
  }
}

} // namespace layers
} // namespace rcnn
