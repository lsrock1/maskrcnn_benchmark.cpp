#pragma once
#include <torch/torch.h>


namespace rcnn {
namespace layers {

template<typename T>
inline torch::Tensor cat(T inputs, int dim = 0) {
  if (inputs.size() == 1) {
    return inputs[0];
  }
  return torch::cat(inputs, dim);
}

} // namespace layers
} // namespace rcnn
