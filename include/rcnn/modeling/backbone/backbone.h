#pragma once
#include "resnet.h"
#include "fpn.h"


namespace rcnn{
namespace modeling{
  
  template<typename Backbone, typename FPNType>
  class BackboneImpl : public torch::nn::Module{
    public:
      BackboneImpl(Backbone body);
      std::vector<torch::Tensor> forward(torch::Tensor x);
    
    private:
      Backbone body_{nullptr};
      FPNType fpn_{nullptr};
  };

  template class BackboneImpl<ResNet, FPNLastMaxPool>;
  using ResBackboneImpl = BackboneImpl<ResNet, FPNLastMaxPool>;

  TORCH_MODULE(ResBackbone);

  torch::nn::Sequential BuildBackBone();
}
}