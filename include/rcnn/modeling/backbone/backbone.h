#pragma once
#include "resnet.h"
#include "fpn.h"


namespace rcnn{
namespace modeling{

class BackboneImpl : torch::nn::Module{

public:
  explicit BackboneImpl(torch::nn::Sequential backbone, int64_t out_channels);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  int64_t get_out_channels();

private:
  torch::nn::Sequential backbone_;
  int64_t out_channels_;
};

TORCH_MODULE(Backbone);

Backbone BuildResnetBackbone();
Backbone BuildResnetFPNBackbone();
Backbone BuildBackbone();

}
}