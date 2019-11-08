#pragma once
#include <torch/torch.h>

namespace rcnn {
namespace layers {

class FrozenBatchNorm2dImpl : public torch::nn::Module {
  
public:
  FrozenBatchNorm2dImpl(int64_t dimension);
  torch::Tensor forward(torch::Tensor x);
  std::shared_ptr<FrozenBatchNorm2dImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

private:
  torch::Tensor weight, bias, mean, var;
  
};

TORCH_MODULE(FrozenBatchNorm2d);

inline FrozenBatchNorm2d BatchNorm(int64_t channels) {
  return FrozenBatchNorm2d(channels);
}

} // namespace layers
} // namespace rcnn
