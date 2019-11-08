#pragma once
#include <torch/torch.h>

namespace rcnn {
namespace layers {

inline torch::Tensor smooth_l1_loss(torch::Tensor input, torch::Tensor target, float beta=1. / 9, bool size_average = true) {
  torch::Tensor n = torch::abs(input - target);
  auto cond = n < beta;
  cond = cond.to(torch::kF32);

  torch::Tensor loss = cond * ((0.5 * n).pow(2) / beta) + (1 - cond) * (n - 0.5 * beta);
  if (size_average) {
    return loss.mean();
  }
  
  return loss.sum(); 
}

} // namespace layers
} // namespace rcnn
