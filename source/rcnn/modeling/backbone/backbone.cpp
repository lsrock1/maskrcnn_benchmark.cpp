#include "backbone/backbone.h"
#include "make_layers.h"
#include "registry.h"


namespace rcnn{
namespace modeling{

torch::nn::Sequential BuildResnetBackbone(){
  torch::nn::Sequential model;
  rcnn::config::CFGS backbone_name = rcnn::config::GetCFG<rcnn::config::CFGS>({"MODEL", "BACKBONE", "CONV_BODY"});
  auto body = ResNet(backbone_name.get());
  model->push_back(body);
  return model;
}

torch::nn::Sequential BuildResnetFPNBackbone(){
  torch::nn::Sequential model;
  rcnn::config::CFGS backbone_name = rcnn::config::GetCFG<rcnn::config::CFGS>({"MODEL", "BACKBONE", "CONV_BODY"});
  auto body = ResNet(backbone_name.get());
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
  torch::nn::Sequential model = rcnn::utils::BACKBONES(name.get())();
  return model;
}

}
}