#pragma once
#include "resnet.h"
#include "fpn.h"
#include <deque>

namespace rcnn{
namespace modeling{
  
  template<typename Backbone>
  class BackboneImpl : public torch::nn::Module{
    public:
      BackboneImpl(Backbone& body);
      std::deque<torch::Tensor> forward(torch::Tensor& x);
    
    private:
      Backbone body_{nullptr};
      FPNLastMaxPool fpn_{nullptr};
  };

  template class BackboneImpl<ResNet>;
  using ResBackboneImpl = BackboneImpl<ResNet>;

  TORCH_MODULE(ResBackbone);

//   Model select_model(std::string model_name);

//   Model build_backbone(YAML::Node cfg);
}
}