#pragma once
#include "resnet.h"
#include "fpn.h"
#include "yaml-cpp/yaml.h"

namespace rcnn{
namespace modeling{
  
  template<typename Backbone>
  class ModelImpl : public torch::nn::Module{
    public:
      ModelImpl(Backbone body, FPN fpn);
      ModelImpl(Backbone body);
      torch::Tensor forward(torch::Tensor x);
    
    private:
      Backbone body_{nullptr};
      FPN fpn_{nullptr};
  };

  TORCH_MODULE(Model);

  Model select_model(std::string model_name);

  Model build_backbone(YAML::Node cfg);
}
}