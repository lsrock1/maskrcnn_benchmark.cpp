#pragma once
#include <torch/torch.h>

namespace rcnn{
namespace layers{

class FrozenBatchNorm2dImpl : public torch::nn::Module {
  public:
    FrozenBatchNorm2dImpl(int64_t dimension);
    torch::Tensor forward(torch::Tensor x);
  
  private:
    torch::Tensor weight, bias, mean, var;
};

TORCH_MODULE(FrozenBatchNorm2d);

FrozenBatchNorm2d BatchNorm(int64_t channels);

}//layers
}//rcnn