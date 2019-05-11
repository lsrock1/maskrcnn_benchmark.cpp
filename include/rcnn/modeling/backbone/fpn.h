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
      
    public:
      FPNImpl(const bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels);
      std::deque<torch::Tensor> forward(std::vector<torch::Tensor>& x);
  };

  TORCH_MODULE(FPN);
  
  class FPNLastMaxPoolImpl : public torch::nn::Module{
    private:
      FPN fpn_;

    public:
      FPNLastMaxPoolImpl(const bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels);
      std::deque<torch::Tensor> forward(std::vector<torch::Tensor>& x);
  };

  TORCH_MODULE(FPNLastMaxPool);
}
}