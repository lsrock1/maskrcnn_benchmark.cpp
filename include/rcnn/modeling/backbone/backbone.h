#pragma once
#include "resnet.h"
#include "fpn.h"

namespace rcnn{
namespace modeling{
  
  template<typename Backbone>
  class BackboneImpl : public torch::nn::Module{
    public:
      BackboneImpl(Backbone body, FPN fpn);
      BackboneImpl(Backbone body);
      torch::Tensor forward(torch::Tensor x);
    
    private:
      Backbone body_{nullptr};
      FPN fpn_{nullptr};
  };
  

//   Model select_model(std::string model_name);

//   Model build_backbone(YAML::Node cfg);
}
}