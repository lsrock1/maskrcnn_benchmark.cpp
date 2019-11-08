#pragma once
#include <torch/torch.h>

using namespace torch::autograd;

namespace rcnn {
namespace layers {

class _NewEmptyTensorOp : public torch::autograd::Function<_NewEmptyTensorOp> {

public:
  static inline Variable forward(AutogradContext *ctx, const Variable& x, std::vector<int64_t> new_shape) {
    ctx->saved_data["shape"] = x.sizes();
    return torch::empty(new_shape, x.options());
  }

  static inline variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
    auto shape = ctx->saved_data["shape"].toIntListRef();
    const Variable grad_output = grad_outputs[0];
    return variable_list{_NewEmptyTensorOp::apply(grad_output, shape.vec()), Variable()};
  }

};

class Conv2dImpl : public torch::nn::Conv2dImpl {

public:
  Conv2dImpl(torch::nn::Conv2dOptions conv2dOptions): torch::nn::Conv2dImpl(conv2dOptions){};
  torch::Tensor forward(const torch::Tensor& input);

};

TORCH_MODULE(Conv2d);

torch::Tensor interpolate(torch::Tensor input, torch::IntArrayRef size/* , float scale_factor, std::string mode, bool align_corners*/);

} // namespace layers
} // namespace rcnn
