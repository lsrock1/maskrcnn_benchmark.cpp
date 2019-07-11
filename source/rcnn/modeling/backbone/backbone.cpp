#include "backbone/backbone.h"
#include "backbone/resnet.h"
#include "backbone/vovnet.h"
#include "backbone/fpn.h"

#include <make_layers.h>
#include <registry.h>
#include <defaults.h>


namespace rcnn{
namespace modeling{

BackboneImpl::BackboneImpl(torch::nn::Sequential backbone, int64_t out_channels) : backbone_(register_module("backbone", backbone)), out_channels_(out_channels){};

std::vector<torch::Tensor> BackboneImpl::forward(torch::Tensor x){
  return backbone_->forward<std::vector<torch::Tensor>>(x);
}

int64_t BackboneImpl::get_out_channels(){
  return out_channels_;
}

std::shared_ptr<BackboneImpl> BackboneImpl::clone(torch::optional<torch::Device> device) const{
  torch::NoGradGuard no_grad;
  torch::nn::Sequential backbone_copy;
  auto backbone_clone = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(backbone_->clone());
  static_cast<torch::nn::SequentialImpl&>(*backbone_copy) = std::move(*backbone_clone);
  std::shared_ptr<BackboneImpl> copy = std::make_shared<BackboneImpl>(backbone_copy, out_channels_);
  if(device.has_value())
    copy->to(device.value());
  return copy;
}

Backbone BuildResnetBackbone(){
  torch::nn::Sequential model;
  auto body = ResNet();
  model->push_back(body);
  auto backbone = Backbone(model, rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "BACKBONE_OUT_CHANNELS"}));
  return backbone;
}

Backbone BuildResnetFPNBackbone(){
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
  auto backbone = Backbone(model, out_channels);
  return backbone;
}

Backbone BuildVoVNetFPNBackbone(){
  torch::nn::Sequential model;
  VoVNet body = VoVNet();
  int64_t in_channels_stage = rcnn::config::GetCFG<int64_t>({"MODEL", "VOVNET", "OUT_CHANNELS"});
  int64_t out_channels = rcnn::config::GetCFG<int64_t>({"MODEL", "VOVNET", "BACKBONE_OUT_CHANNELS"});
  model->push_back(body);
  model->push_back(
    FPNLastMaxPool(
      rcnn::config::GetCFG<bool>({"MODEL", "FPN", "USE_RELU"}), 
      std::vector<int64_t>{
        in_channels_stage,
        in_channels_stage * 2,
        in_channels_stage * 3,
        in_channels_stage * 4
      }, 
      out_channels, 
      rcnn::layers::ConvWithKaimingUniform
    )
  );
  auto backbone = Backbone(model, out_channels);
  return backbone;
}
  
Backbone BuildBackbone(){
  std::string name = rcnn::config::GetCFG<std::string>({"MODEL", "BACKBONE", "CONV_BODY"});
  rcnn::registry::backbone build_function = rcnn::registry::BACKBONES(name);
  Backbone model = build_function();
  return model;
}

}
}