#include "backbone/resnet.h"
#include "defaults.h"
#include <cassert>
#include <cmath>


namespace rcnn {
namespace modeling {

ResNetImpl::StageSpec::StageSpec(int index, int block_count, bool return_features)
  :index_(index),
   block_count_(block_count),
   return_features_(return_features){}

ResNetImpl::ResNetImpl() {
  std::string name = rcnn::config::GetCFG<std::string>({"MODEL", "BACKBONE", "CONV_BODY"});
  std::vector<StageSpec> stage_specs = rcnn::registry::STAGE_SPECS(name);
  stem_ = register_module("stem", BaseStem());
  int64_t num_groups = rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "NUM_GROUPS"});
  int64_t width_per_group = rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "WIDTH_PER_GROUP"});
  int64_t in_channels = rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "STEM_OUT_CHANNELS"});
  int64_t stage2_bottleneck_channels = num_groups * width_per_group;
  int64_t stage2_out_channels = rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "RES2_OUT_CHANNELS"});

  for (auto& stage_spec: stage_specs) {
    std::string name = "layer" + std::to_string(stage_spec.index_);
    int64_t stage2_relative_factor = pow((int64_t) 2, (int64_t) (stage_spec.index_ - 1));
    int64_t bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor;
    int64_t out_channels = stage2_out_channels * stage2_relative_factor;
    stages_.push_back(
      register_module(
        name,
        MakeStage(
          in_channels,
          bottleneck_channels,
          out_channels,
          stage_spec.block_count_,
          num_groups,
          rcnn::config::GetCFG<bool>({"MODEL", "RESNETS", "STRIDE_IN_1X1"}),
          ((int64_t) (stage_spec.index_ > 1)) + 1
        )
      )//register_module
    );//push_back
    in_channels = out_channels;
    return_features_.push_back(stage_spec.return_features_);
  }
  freeze_backbone(rcnn::config::GetCFG<int64_t>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));
}

std::shared_ptr<ResNetImpl> ResNetImpl::clone(torch::optional<torch::Device> device) const {
  torch::NoGradGuard no_grad;
  std::shared_ptr<ResNetImpl> copy = std::make_shared<ResNetImpl>();
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

std::vector<torch::Tensor> ResNetImpl::forward(torch::Tensor x) {
  std::vector<torch::Tensor> results;
  x = stem_(x);
  for (size_t i = 0; i < stages_.size(); ++i) {
    x = stages_.at(i)->forward(x);
    if (return_features_.at(i))
      results.push_back(x);
  }
  return results;
}

void ResNetImpl::freeze_backbone(int freeze_at) {
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

    if (4 < freeze_at) {
      std::vector<torch::Tensor> params = stages_.at(3)->parameters();
      for (auto& i: params)
        i.set_requires_grad(false);
    }
  }
}

torch::nn::Sequential MakeStage(/*transformation_module, */
  int64_t in_channels, 
  int64_t bottleneck_channels, 
  int64_t out_channels, 
  int64_t block_count, 
  int64_t num_groups, 
  bool stride_in_1x1, 
  int64_t first_stride, 
  int64_t dilation){
  torch::nn::Sequential blocks;
  int64_t stride = first_stride;

  for (int64_t i = 0; i < block_count; ++i) {
    blocks->push_back(
      Bottleneck(
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation
      )
    );
    stride = 1;
    in_channels = out_channels;
  }

  return blocks;
}

BottleneckImpl::BottleneckImpl(
  int64_t in_channels, 
  int64_t bottleneck_channels, 
  int64_t out_channels, 
  int64_t num_groups, 
  bool stride_in_1x1, 
  int64_t stride, 
  int64_t dilation/*, norm_func , dcn_config*/) {
    
  if (in_channels != out_channels) {
    int64_t down_stride = (dilation == 1 ? stride : 1);
    downsample_ = register_module("downsample", torch::nn::Sequential(
      rcnn::layers::Conv2d(
        torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(down_stride).with_bias(false)
      ),
      rcnn::layers::FrozenBatchNorm2d(out_channels)
    )
    );
    for (auto& param : downsample_->named_parameters()) {
      if (param.key().find("weight") != std::string::npos && param.key().find("conv") != std::string::npos) {
        torch::nn::init::kaiming_uniform_(param.value(), 1);
      }
    }
  }

  if (dilation > 1)
    stride = 1;

  int64_t stride_1x1, stride_3x3;
  if (stride_in_1x1) {
    stride_1x1 = stride;
    stride_3x3 = 1;
  }
  else {
    stride_1x1 = 1;
    stride_3x3 = stride;
  }

  conv1_ = register_module(
    "conv1", rcnn::layers::Conv2d(torch::nn::Conv2dOptions(in_channels, bottleneck_channels, 1).stride(stride_1x1).with_bias(false))
  );
  bn1_ = register_module("bn1", rcnn::layers::FrozenBatchNorm2d(bottleneck_channels));

  conv2_ = register_module("conv2", 
    rcnn::layers::Conv2d(
      torch::nn::Conv2dOptions(bottleneck_channels, bottleneck_channels, 3)
        .stride(stride_3x3)
        .padding(dilation)
        .with_bias(false)
        .groups(num_groups)
        .dilation(dilation)
    )
  );
  
  bn2_ = register_module("bn2", rcnn::layers::FrozenBatchNorm2d(bottleneck_channels));

  conv3_ = register_module("conv3", rcnn::layers::Conv2d(torch::nn::Conv2dOptions(bottleneck_channels, out_channels, 1).with_bias(false)));
  bn3_ = register_module("bn3", rcnn::layers::FrozenBatchNorm2d(out_channels));
  torch::nn::init::kaiming_uniform_(conv1_->weight, 1);
  torch::nn::init::kaiming_uniform_(conv2_->weight, 1);
  torch::nn::init::kaiming_uniform_(conv3_->weight, 1);
};

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
  torch::Tensor identity;
  if (downsample_) {
    identity = downsample_->forward(x);
  }
  else {
    identity = x;
  }

  x = bn1_->forward(conv1_->forward(x)).relu_();
  x = bn2_->forward(conv2_->forward(x)).relu_();
  x = bn3_->forward(conv3_->forward(x));
  x += identity;
  return x.relu_();
}

BaseStemImpl::BaseStemImpl() {
  int64_t out_channels = rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "STEM_OUT_CHANNELS"});
  conv1_ = register_module(
    "conv1", rcnn::layers::Conv2d(torch::nn::Conv2dOptions(3, out_channels, 7).stride(2).padding(3).with_bias(false))
  );

  bn1_ = register_module("bn1", rcnn::layers::FrozenBatchNorm2d(out_channels));
  torch::nn::init::kaiming_uniform_(conv1_->weight, 1);
}

torch::Tensor BaseStemImpl::forward(torch::Tensor& x) {
  x = bn1_->forward(conv1_->forward(x)).relu_();
  x = torch::max_pool2d(x, 3, 2, 1);
  return x;
}

ResNetHeadImpl::ResNetHeadImpl(
  std::vector<ResNetImpl::StageSpec> stages, 
  int64_t num_groups, 
  int64_t width_per_groups, 
  bool stride_in_1x1, 
  int64_t stride_init, 
  int64_t res2_out_channels, 
  int64_t dilation) {

  int64_t stage2_relative_factor = pow((int64_t)2, (int64_t) (stages.at(0).index_ - 1));
  int64_t stage2_bottleneck_channels = num_groups * width_per_groups;
  int64_t out_channels = res2_out_channels * stage2_relative_factor;
  int64_t in_channels = (int64_t) out_channels / 2;
  int64_t bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor;

  int64_t stride = stride_init;

  for (auto& stage_spec: stages) {
    std::string name = "layer" + std::to_string(stage_spec.index_);
    if (stride == 0)
      stride = ((int) (stage_spec.index_ > 1)) + 1;
    stages_.push_back(
      register_module(
        name,
        MakeStage(
          in_channels,
          bottleneck_channels,
          out_channels,
          stage_spec.block_count_,
          num_groups,
          stride_in_1x1,
          stride,
          dilation
        )//make stage
      )//register module
    );//push back
    stride = 0;
  }
  out_channels_ = out_channels;
}

int64_t ResNetHeadImpl::out_channels() const {
  return out_channels_;
}

torch::Tensor ResNetHeadImpl::forward(torch::Tensor x) {
  for (auto& stage: stages_)
    x = stage->forward(x);

  return x;
}

} // namespace modeling

namespace registry {

std::vector<rcnn::modeling::ResNetImpl::StageSpec> STAGE_SPECS(std::string name) {
  std::map<std::string, std::vector<rcnn::modeling::ResNetImpl::StageSpec>> _STAGE_SPECS{
    {"R-50-C4", std::vector<rcnn::modeling::ResNetImpl::StageSpec>{
      rcnn::modeling::ResNetImpl::StageSpec(1, 3, false),
      rcnn::modeling::ResNetImpl::StageSpec(2, 4, false),
      rcnn::modeling::ResNetImpl::StageSpec(3, 6, true)
    }},
    {"R-50-C5", std::vector<rcnn::modeling::ResNetImpl::StageSpec>{
      rcnn::modeling::ResNetImpl::StageSpec(1, 3, false),
      rcnn::modeling::ResNetImpl::StageSpec(2, 4, false),
      rcnn::modeling::ResNetImpl::StageSpec(3, 6, false),
      rcnn::modeling::ResNetImpl::StageSpec(4, 3, true)
    }},
    {"R-101-C4", std::vector<rcnn::modeling::ResNetImpl::StageSpec>{
      rcnn::modeling::ResNetImpl::StageSpec(1, 3, false),
      rcnn::modeling::ResNetImpl::StageSpec(2, 4, false),
      rcnn::modeling::ResNetImpl::StageSpec(3, 23, true)
    }},
    {"R-101-C5", std::vector<rcnn::modeling::ResNetImpl::StageSpec>{
      rcnn::modeling::ResNetImpl::StageSpec(1, 3, false),
      rcnn::modeling::ResNetImpl::StageSpec(2, 4, false),
      rcnn::modeling::ResNetImpl::StageSpec(3, 23, false),
      rcnn::modeling::ResNetImpl::StageSpec(4, 3, true)
    }},
    {"R-50-FPN", std::vector<rcnn::modeling::ResNetImpl::StageSpec>{
      rcnn::modeling::ResNetImpl::StageSpec(1, 3, true),
      rcnn::modeling::ResNetImpl::StageSpec(2, 4, true),
      rcnn::modeling::ResNetImpl::StageSpec(3, 6, true),
      rcnn::modeling::ResNetImpl::StageSpec(4, 3, true)
    }},
    {"R-101-FPN", std::vector<rcnn::modeling::ResNetImpl::StageSpec>{
      rcnn::modeling::ResNetImpl::StageSpec(1, 3, true),
      rcnn::modeling::ResNetImpl::StageSpec(2, 4, true),
      rcnn::modeling::ResNetImpl::StageSpec(3, 23, true),
      rcnn::modeling::ResNetImpl::StageSpec(4, 3, true)
    }},
    {"R-152-FPN", std::vector<rcnn::modeling::ResNetImpl::StageSpec>{
      rcnn::modeling::ResNetImpl::StageSpec(1, 3, true),
      rcnn::modeling::ResNetImpl::StageSpec(2, 8, true),
      rcnn::modeling::ResNetImpl::StageSpec(3, 36, true),
      rcnn::modeling::ResNetImpl::StageSpec(4, 3, true)
    }}
  };
  assert(_STAGE_SPECS.count(name));
  return _STAGE_SPECS.find(name)->second;
}

} // namespace registry
} // namespace rcnn
