#include "backbone/vovnet.h"
#include <cassert>
#include <iostream>

#include <defaults.h>

namespace rcnn {
namespace modeling {

OSA_moduleImpl::OSA_moduleImpl(int64_t in_channels, 
                               int64_t stage_channels, 
                               int64_t concat_channels, 
                               int layers_per_block, 
                               std::string module_name, 
                               bool identity)
                               :identity_(identity) {
  int64_t channels = in_channels;

  for (int i = 0; i < layers_per_block; ++i) {
    torch::nn::Sequential seq;
    conv3x3(seq, channels, stage_channels, module_name, std::to_string(i));
    layers_.push_back(register_module("layers_" + std::to_string(i), seq));
    channels = stage_channels;
  }
  channels = in_channels + layers_per_block * stage_channels;
  torch::nn::Sequential concat;
  conv1x1(concat, channels, concat_channels, module_name, "concat");
  concat_ = register_module("concat", concat);
}

torch::Tensor OSA_moduleImpl::forward(torch::Tensor x) {
  torch::Tensor identity_feat = x;
  std::vector<torch::Tensor> output{x};

  for (auto& layer : layers_) {
    x = layer->forward(x);
    output.push_back(x);
  }

  x = torch::cat(output, 1);
  x = concat_->forward(x);
  
  if (identity_)
    x += identity_feat;

  return x;
}

OSA_stageImpl::OSA_stageImpl(int64_t in_channels, 
                            int64_t stage_channels, 
                            int64_t concat_channels, 
                            int blocks_per_stage, 
                            int layers_per_block, 
                            int stage_num) {
  if (stage_num != 2)
    modules->push_back("Pooling", torch::nn::Functional(torch::max_pool2d, /*kernel_size =*/3, /*stride =*/2, /*padding =*/0, /*dilation =*/1, /*ceil_mode =*/true));
  
  std::string module_name = "OSA" + std::to_string(stage_num) + "_1";
  modules->push_back(std::string(module_name), std::move(OSA_module(in_channels, stage_channels, concat_channels, layers_per_block, module_name)));

  for (int i = 0; i < blocks_per_stage - 1; ++i) {
    module_name = "OSA" + std::to_string(stage_num) + "_" + std::to_string(i + 2);
    modules->push_back(std::string(module_name), std::move(OSA_module(concat_channels, stage_channels, concat_channels, layers_per_block, module_name, true)));
  }
  register_module("stage", modules);
}

torch::Tensor OSA_stageImpl::forward(torch::Tensor x) {
  return modules->forward(x);
}

VoVNetImpl::StageSpec::StageSpec(std::vector<int> stage_channels, 
                                 std::vector<int> concat_channels, 
                                 int layers_per_block, 
                                 std::vector<int> blocks_per_stage)
                                :stage_channels_(stage_channels),
                                 concat_channels_(concat_channels),
                                 layers_per_block_(layers_per_block),
                                 blocks_per_stage_(blocks_per_stage){}

VoVNetImpl::VoVNetImpl() {
  std::string name = rcnn::config::GetCFG<std::string>({"MODEL", "BACKBONE", "CONV_BODY"});
  StageSpec stage_specs = rcnn::registry::STAGE_SPECS_VoVNet(name);
  auto concat_channels = stage_specs.concat_channels_;
  auto stage_channels = stage_specs.stage_channels_;
  int layers_per_block = stage_specs.layers_per_block_;
  auto blocks_per_stage = stage_specs.blocks_per_stage_;

  conv3x3(stem_, 3, 64, "stem", "1", 2);
  conv3x3(stem_, 64, 64, "stem", "2", 1);
  conv3x3(stem_, 64, 128, "stem", "3", 2);
  stem_ = register_module("stem", stem_);
  std::vector<int64_t> in_channels_list{128};
  in_channels_list.insert(in_channels_list.end(), concat_channels.begin(), concat_channels.end()-1);
  for (decltype(in_channels_list.size()) i = 0; i < in_channels_list.size(); ++i) {
    std::string name = "stage" + std::to_string(i + 2);
    stages_.push_back(register_module(name, OSA_stage(in_channels_list.at(i), stage_channels.at(i), concat_channels.at(i), blocks_per_stage.at(i), layers_per_block, i + 2)));
  }
  initialize_weights();
  freeze_backbone(rcnn::config::GetCFG<int64_t>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));
}

std::vector<torch::Tensor> VoVNetImpl::forward(torch::Tensor x) {
  x = stem_->forward(x);
  std::vector<torch::Tensor> outputs;
  for (auto& stage : stages_) {
    x = stage->forward(x);
    outputs.push_back(x);
  }
  return outputs;
}

std::shared_ptr<VoVNetImpl> VoVNetImpl::clone(torch::optional<torch::Device> device) const {
  torch::NoGradGuard no_grad;
  std::shared_ptr<VoVNetImpl> copy = std::make_shared<VoVNetImpl>();
  auto named_params = named_parameters();
  auto named_bufs = named_buffers();
  for (auto& i : copy->named_parameters()) {
    i.value().copy_(named_params[i.key()]);
  }
  for (auto& i : copy->named_buffers()) {
    i.value().copy_(named_bufs[i.key()]);
  }
  if (device.has_value())
    copy->to(device.value());
  return copy;
}

void VoVNetImpl::initialize_weights() {
  for (auto& i : named_parameters()){
    if (i.key().find("conv") != std::string::npos)
      torch::nn::init::kaiming_uniform_(i.value());
  }
}

void VoVNetImpl::freeze_backbone(int freeze_at) {
  if (freeze_at < 0) {
    return;
  }
  else {
    if (0 < freeze_at) {
      std::vector<torch::Tensor> params = stem_->parameters();
      for (auto& i: params)
        i.set_requires_grad(false);
    }

    if (1 < freeze_at) {
      std::vector<torch::Tensor> params = stages_.at(0)->parameters();
      for (auto& i: params)
        i.set_requires_grad(false);
    }

    if (2 < freeze_at) {
      std::vector<torch::Tensor> params = stages_.at(1)->parameters();
      for (auto& i: params)
        i.set_requires_grad(false);
    }

    if (3 < freeze_at) {
      std::vector<torch::Tensor> params = stages_.at(2)->parameters();
      for (auto& i: params)
        i.set_requires_grad(false);
    }

    if (4 < freeze_at){
      std::vector<torch::Tensor> params = stages_.at(3)->parameters();
      for (auto& i: params)
        i.set_requires_grad(false);
    }
  }
}

void conv3x3(torch::nn::Sequential& seq,
             int64_t in_channels, 
             int64_t out_channels, 
             std::string module_name, 
             std::string postfix, 
             int64_t stride, 
             int64_t groups, 
             int64_t kernel_size, 
             int64_t padding) {
  seq->push_back(module_name + "_" + postfix + "/conv", rcnn::layers::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding).groups(groups).with_bias(false)));
  seq->push_back(module_name + "_" + postfix + "/norm", rcnn::layers::FrozenBatchNorm2d(out_channels));
  seq->push_back(module_name + "_" + postfix + "/relu", torch::nn::Functional(torch::relu));
}

void conv1x1(torch::nn::Sequential& seq,
             int64_t in_channels, 
             int64_t out_channels, 
             std::string module_name, 
             std::string postfix, 
             int64_t stride, 
             int64_t groups, 
             int64_t kernel_size, 
             int64_t padding) {
  seq->push_back(module_name + "_" + postfix + "/conv", rcnn::layers::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding).groups(groups).with_bias(false)));
  seq->push_back(module_name + "_" + postfix + "/norm", rcnn::layers::FrozenBatchNorm2d(out_channels));
  seq->push_back(module_name + "_" + postfix + "/relu", torch::nn::Functional(torch::relu));
}
} // namespace modeling

namespace registry {

// const _STEM_MODULES
rcnn::modeling::VoVNetImpl::StageSpec STAGE_SPECS_VoVNet(std::string name) {
  std::map<std::string, rcnn::modeling::VoVNetImpl::StageSpec> _STAGE_SPECS{
    {std::string("V-57-FPN"), rcnn::modeling::VoVNetImpl::StageSpec(std::vector<int>{128, 160, 192, 224}, std::vector<int>{256, 512, 768, 1024}, 5, std::vector<int>{1, 1, 4, 3})},
    {std::string("V-39-FPN"), rcnn::modeling::VoVNetImpl::StageSpec(std::vector<int>{128, 160, 192, 224}, std::vector<int>{256, 512, 768, 1024}, 5, std::vector<int>{1, 1, 2, 2})}
  };

  assert(_STAGE_SPECS.count(name));
  return _STAGE_SPECS.find(name)->second;
}

} // namespace registry
} // namespace rcnn
