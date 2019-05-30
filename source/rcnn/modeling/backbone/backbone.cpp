#include "backbone/backbone.h"
#include "make_layers.h"
#include "registry.h"
#include "defaults.h"
#include <iostream>


namespace rcnn{
namespace modeling{

torch::nn::Sequential BuildResnetBackbone(){
  torch::nn::Sequential model;
  auto body = ResNet();
  model->push_back(body);
  return model;
}

torch::nn::Sequential BuildResnetFPNBackbone(){
  torch::nn::Sequential model;
  ResNet body = ResNet();
  int64_t in_channels_stage2 = rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "RES2_OUT_CHANNELS"});
  int64_t out_channels = rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "BACKBONE_OUT_CHANNELS"});
  model->push_back(body);
  model->push_back(
    FPNLastMaxPool(
      rcnn::config::GetCFG<bool>({"MODEL", "FPN", "USE_RELU"}), 
      std::vector<int64_t>{
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8
      }, 
      out_channels, 
      rcnn::layers::ConvWithKaimingUniform
    )
  );
  return model;
}
  
torch::nn::Sequential BuildBackbone(){
  rcnn::config::CFGS name = rcnn::config::GetCFG<rcnn::config::CFGS>({"MODEL", "BACKBONE", "CONV_BODY"});
  rcnn::registry::backbone build_function = rcnn::registry::BACKBONES(name.get());
  torch::nn::Sequential model = build_function();
  return model;
}

}
}