#pragma once
#include <torch/torch.h>
#include <vector>
#include <deque>
#include "conv2d.h"
#include "make_layers.h"

namespace rcnn{
namespace modeling{
  class FPNImpl : public torch::nn::Module{
    private:
      std::vector<torch::nn::Sequential> inner_blocks_;
      std::vector<torch::nn::Sequential> layer_blocks_;
      torch::nn::Sequential inner_block1_{nullptr}, inner_block2_{nullptr}, inner_block3_{nullptr}, inner_block4_{nullptr};
      torch::nn::Sequential layer_block1_{nullptr}, layer_block2_{nullptr}, layer_block3_{nullptr}, layer_block4_{nullptr};
      
    public:
      FPNImpl(const bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels);
      std::deque<torch::Tensor> forward(std::vector<torch::Tensor>& x);
  };

  TORCH_MODULE(FPN);

  class LastLevelMaxPoolImpl : public torch::nn::Module{
    public:
      torch::Tensor forward(torch::Tensor& x);
  };

  TORCH_MODULE(LastLevelMaxPool);
  
  class FPNLastMaxPoolImpl : public torch::nn::Module{
    private:
      LastLevelMaxPool last_level_;
      FPN fpn_;

    public:
      FPNLastMaxPoolImpl(const bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels);
      std::deque<torch::Tensor> forward(std::vector<torch::Tensor>& x);
  };

  TORCH_MODULE(FPNLastMaxPool);
}
}