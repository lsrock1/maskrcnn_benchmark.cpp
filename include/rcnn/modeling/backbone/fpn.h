#pragma once
#include <torch/torch.h>
#include <vector>


namespace rcnn{
namespace modeling{

using ConvFunction = torch::nn::Sequential (*) (bool, int64_t, int64_t, int64_t, int64_t, int64_t);

class FPNImpl : public torch::nn::Module{
    
public:
  FPNImpl(const bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels, ConvFunction conv_block);
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor>& x);
  std::shared_ptr<FPNImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

  const bool use_relu_;
  const std::vector<int64_t> in_channels_list_;
  const int64_t out_channels_;
  ConvFunction conv_block_;

private:
  std::vector<torch::nn::Sequential> inner_blocks_;
  std::vector<torch::nn::Sequential> layer_blocks_;
};

TORCH_MODULE(FPN);

class FPNLastMaxPoolImpl : public torch::nn::Module{

public:
  FPNLastMaxPoolImpl(const bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels, ConvFunction conv_block);
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);
  std::shared_ptr<FPNLastMaxPoolImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

private:
  FPN fpn_;
};

TORCH_MODULE(FPNLastMaxPool);

}
}