#include "batch_norm.h"

namespace rcnn{
namespace layers{

FrozenBatchNorm2dImpl::FrozenBatchNorm2dImpl(int64_t dimension)
  : weight(register_buffer("weight", torch::ones(dimension))),
  bias(register_buffer("bias", torch::zeros(dimension))),
  mean(register_buffer("running_mean", torch::zeros(dimension))),
  var(register_buffer("running_var", torch::ones(dimension))){};

torch::Tensor FrozenBatchNorm2dImpl::forward(torch::Tensor x){
    // TODO INTEGRATION
  torch::Tensor scale_n = weight * var.rsqrt();
  torch::Tensor bias_n = bias - mean * scale_n;
  scale_n = scale_n.reshape({1, -1, 1, 1});
  bias_n = bias_n.reshape({1, -1, 1, 1});
  return x * scale_n + bias_n;
};

FrozenBatchNorm2d BatchNorm(int64_t channels){
  return FrozenBatchNorm2d(channels);
}

}//layers
}//rcnn