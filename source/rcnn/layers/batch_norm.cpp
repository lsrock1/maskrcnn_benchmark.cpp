#include <batch_norm.h>

namespace rcnn {
namespace layers {

FrozenBatchNorm2dImpl::FrozenBatchNorm2dImpl(int64_t dimension)
  : weight(register_buffer("weight", torch::ones(dimension))),
    bias(register_buffer("bias", torch::zeros(dimension))),
    mean(register_buffer("running_mean", torch::zeros(dimension))),
    var(register_buffer("running_var", torch::ones(dimension))) {};

std::shared_ptr<torch::nn::Module> FrozenBatchNorm2dImpl::clone(const torch::optional<torch::Device>& device) const {
  torch::NoGradGuard no_grad;
  std::shared_ptr<FrozenBatchNorm2dImpl> copy = std::make_shared<FrozenBatchNorm2dImpl>(mean.size(0));
  auto named_bufs = named_buffers();
  
  for (auto& i : copy->named_buffers()) {
    i.value().copy_(named_bufs[i.key()]);
  }
  
  if (device.has_value()) {
    copy->to(device.value());
  }

  return copy;
}

torch::Tensor FrozenBatchNorm2dImpl::forward(torch::Tensor x) {
    // TODO INTEGRATION
  torch::Tensor scale_n = weight * var.rsqrt();
  torch::Tensor bias_n = bias - mean * scale_n;
  scale_n = scale_n.reshape({1, -1, 1, 1});
  bias_n = bias_n.reshape({1, -1, 1, 1});
  return x * scale_n + bias_n;
};

} // namespace layers
} // namespace rcnn
