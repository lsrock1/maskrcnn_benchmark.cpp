#include "backbone/resnet.h"


namespace rcnn{
namespace modeling{

ResNetImpl::StageSpec::StageSpec(std::string block, std::initializer_list<int64_t> num_layers, int stage_to, bool is_fpn, int64_t groups, int64_t width_per_group, int freeze_at)
                    : num_layers_(num_layers),
                      stage_to_(stage_to),
                      is_fpn_(is_fpn),
                      groups_(groups),
                      width_per_group_(width_per_group),
                      block_(block),
                      freeze_at_(freeze_at)
                      {}

std::string ResNetImpl::StageSpec::get_block(){
  return this->block_;
}

std::initializer_list<int64_t> ResNetImpl::StageSpec::get_num_layers(){
  return this->num_layers_;
}

int ResNetImpl::StageSpec::get_stage_to(){
  return this->stage_to_;
}

bool ResNetImpl::StageSpec::get_is_fpn(){
  return this->is_fpn_;
}

int64_t ResNetImpl::StageSpec::get_groups(){
  return this->groups_;
}

int64_t ResNetImpl::StageSpec::get_width_per_group(){
  return this->width_per_group_;
}

int ResNetImpl::StageSpec::get_freeze_at(){
  return this->freeze_at_;
}

ResNetImpl::ResNetImpl(StageSpec& stage_spec)
            : groups_(stage_spec.get_groups()),
              base_width_(stage_spec.get_width_per_group()),
              block_(stage_spec.get_block()),
              in_planes_(rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "STEM_OUT_CHANNELS"})),
              expansion_(block_.compare("Bottleneck") == 0 ? BottleneckImpl::kExpansion : BasicBlockImpl::kExpansion),
              conv1_(register_module("conv1", rcnn::layers::Conv2d(torch::nn::Conv2dOptions(3, in_planes_, 7).stride(2).padding(3).with_bias(false)))),
              bn1_(register_module("bn1", rcnn::layers::FrozenBatchNorm2d(in_planes_))){
                  is_fpn_ = stage_spec.get_is_fpn();
                  auto it = stage_spec.get_num_layers().begin();
                  layer1_ = register_module("layer1", MakeLayer(rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "RES2_OUT_CHANNELS"})/4, *(it++)));
                  layer2_ = register_module("layer2", MakeLayer(128, *(it++), 2));
                  layer3_ = register_module("layer3", MakeLayer(256, *(it++), 2));
                  if(stage_spec.get_stage_to() > 4){
                    layer4_ = register_module("layer4", MakeLayer(512, *(it++), 2));
                  }

                  initialize();
                  //check freeze over num of layers
                  if(stage_spec.get_freeze_at() > 4 && stage_spec.get_stage_to() <= 4)
                    freeze_backbone(4);
                  else
                    freeze_backbone(stage_spec.get_freeze_at());
              }


bool ResNetImpl::get_is_fpn(){
  return is_fpn_;
}

int64_t ResNetImpl::get_bottom_channels(){
  return rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "RES2_OUT_CHANNELS"});
}

int64_t ResNetImpl::get_out_channels(){
  return rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "BACKBONE_OUT_CHANNELS"});
}

torch::Tensor ResNetImpl::forward(torch::Tensor x){
    x = bn1_->forward(conv1_->forward(x)).relu_();
    x = torch::max_pool2d(x, 3, 2, 1);
    x = layer1_->forward(x);
    x = layer2_->forward(x);
    x = layer3_->forward(x);
    if(layer4_)
        x = layer4_->forward(x);
    return x;
}

std::vector<torch::Tensor> ResNetImpl::forward_fpn(torch::Tensor x){
  std::vector<torch::Tensor> fpn;
  x = bn1_->forward(conv1_->forward(x)).relu_();
  x = torch::max_pool2d(x, 3, 2, 1);
  x = layer1_->forward(x);
  fpn.push_back(x);
  x = layer2_->forward(x);
  fpn.push_back(x);
  x = layer3_->forward(x);
  fpn.push_back(x);
  x = layer4_->forward(x);
  fpn.push_back(x);
  return fpn;
}

void ResNetImpl::freeze_backbone(int freeze_at){
  if(freeze_at < 0){
    return;
  }
  else{
    if(0 < freeze_at){
      std::vector<torch::Tensor> params = conv1_->parameters();
      for(auto i = params.begin(); i != params.end(); ++i)
        (*i).set_requires_grad(false);
    }

    if(1 < freeze_at){
      std::vector<torch::Tensor> params = layer1_->parameters();
      for(auto i = params.begin(); i != params.end(); ++i)
        (*i).set_requires_grad(false);
    }

    if(2 < freeze_at){
      std::vector<torch::Tensor> params = layer2_->parameters();
      for(auto i = params.begin(); i != params.end(); ++i)
        (*i).set_requires_grad(false);
    }

    if(3 < freeze_at){
      std::vector<torch::Tensor> params = layer3_->parameters();
      for(auto i = params.begin(); i != params.end(); ++i)
        (*i).set_requires_grad(false);
    }

    if(4 < freeze_at){
      std::vector<torch::Tensor> params = layer4_->parameters();
      for(auto i = params.begin(); i != params.end(); ++i)
        (*i).set_requires_grad(false);
    }
  }
}

void ResNetImpl::initialize(){
  for(auto &param : this->named_parameters()){
    if(param.key().find("conv") != std::string::npos){
      if(param.key().find("weight") != std::string::npos) {
        torch::nn::init::kaiming_uniform_(param.value(), 1);
      }
    }
  }
}

torch::nn::Sequential ResNetImpl::MakeLayer(int64_t planes, int64_t blocks, int64_t stride){
  torch::nn::Sequential downsample{nullptr};
  if(stride != 1 || in_planes_ != planes * expansion_){
    downsample = torch::nn::Sequential(
      Conv1x1(in_planes_, planes * expansion_, stride),
      rcnn::layers::FrozenBatchNorm2d(planes * expansion_)
    );
  }
  torch::nn::Sequential layers;
  if(block_.compare("Bottleneck") == 0){
    if(downsample){
      layers->push_back(Bottleneck(in_planes_, planes, downsample, stride, groups_, base_width_));
    }
    else{
      layers->push_back(Bottleneck(in_planes_, planes, stride, groups_, base_width_));
    }
  }
  else{
    if(downsample){
      layers->push_back(BasicBlock(in_planes_, planes, downsample, stride, groups_, base_width_));
    }
    else{
      layers->push_back(BasicBlock(in_planes_, planes, stride, groups_, base_width_));
  }
  }
  in_planes_ = planes * expansion_;
  for(int i = 1; i < blocks; ++i){
    if(block_.compare("Bottleneck") == 0){
      layers->push_back(Bottleneck(in_planes_, planes, 1, groups_, base_width_));
    }
    else{
      layers->push_back(BasicBlock(in_planes_, planes, 1, groups_, base_width_));
    }
  }
  return layers;
}

BottleneckImpl::BottleneckImpl(int64_t in_planes, 
                            int64_t out_planes, 
                            torch::nn::Sequential downsample, 
                            int64_t stride, 
                            int64_t groups, 
                            int64_t base_width)
                            : width_(out_planes * (base_width/64.) * groups),
                              conv1_(register_module("conv1", Conv1x1(in_planes, width_))),
                              bn1_(register_module("bn1", rcnn::layers::FrozenBatchNorm2d(width_))),
                              conv2_(register_module("conv2", Conv3x3(width_, width_, stride, groups))),
                              bn2_(register_module("bn2", rcnn::layers::FrozenBatchNorm2d(width_))),
                              conv3_(register_module("conv3", Conv1x1(width_, out_planes * BottleneckImpl::kExpansion))),
                              bn3_(register_module("bn3", rcnn::layers::FrozenBatchNorm2d(out_planes * BottleneckImpl::kExpansion))),
                              downsample_(register_module("downsample", downsample)),
                              stride_(stride){};

BottleneckImpl::BottleneckImpl(int64_t in_planes, 
                            int64_t out_planes, 
                            int64_t stride, 
                            int64_t groups, 
                            int64_t base_width)
                            : width_(out_planes * (base_width/64.) * groups),
                              conv1_(register_module("conv1", Conv1x1(in_planes, width_))),
                              bn1_(register_module("bn1", rcnn::layers::FrozenBatchNorm2d(width_))),
                              conv2_(register_module("conv2", Conv3x3(width_, width_, stride, groups))),
                              bn2_(register_module("bn2", rcnn::layers::FrozenBatchNorm2d(width_))),
                              conv3_(register_module("conv3", Conv1x1(width_, out_planes * BottleneckImpl::kExpansion))),
                              bn3_(register_module("bn3", rcnn::layers::FrozenBatchNorm2d(out_planes * BottleneckImpl::kExpansion))),
                              stride_(stride){};

torch::Tensor BottleneckImpl::forward(torch::Tensor x){
    torch::Tensor identity;
    if(downsample_){
        identity = downsample_->forward(x);
    }
    else{
        identity = x;
    }
    x = bn1_->forward(conv1_->forward(x)).relu_();
    x = bn2_->forward(conv2_->forward(x)).relu_();
    x = bn3_->forward(conv3_->forward(x));
    x += identity;
    return x.relu_();
}

BasicBlockImpl::BasicBlockImpl(int64_t in_planes,
                            int64_t out_planes, 
                            torch::nn::Sequential downsample, 
                            int64_t stride, 
                            int64_t groups, 
                            int64_t base_width/*c++ frontend only has batch norm*/)
                            : conv1_(register_module("conv1", Conv3x3(in_planes, out_planes, stride))),
                              bn1_(register_module("bn1", rcnn::layers::FrozenBatchNorm2d(out_planes))),
                              conv2_(register_module("conv2", Conv3x3(out_planes, out_planes))),
                              bn2_(register_module("bn2", rcnn::layers::FrozenBatchNorm2d(out_planes))),
                              downsample_(register_module("downsample", downsample)),
                              stride_(stride)
                              {};

BasicBlockImpl::BasicBlockImpl(int64_t in_planes,
                            int64_t out_planes,
                            int64_t stride, 
                            int64_t groups, 
                            int64_t base_width/*c++ frontend only has batch norm*/)
                            : conv1_(register_module("conv1", Conv3x3(in_planes, out_planes, stride))),
                              bn1_(register_module("bn1", rcnn::layers::FrozenBatchNorm2d(out_planes))),
                              conv2_(register_module("conv2", Conv3x3(out_planes, out_planes))),
                              bn2_(register_module("bn2", rcnn::layers::FrozenBatchNorm2d(out_planes))),
                              stride_(stride)
                              {};

torch::Tensor BasicBlockImpl::forward(torch::Tensor x){
    torch::Tensor identity;
    if(downsample_){
        identity = downsample_->forward(x);
    }
    else{
        identity = x;
    }
    x = bn1_->forward(conv1_->forward(x)).relu_();
    x = bn2_->forward(conv2_->forward(x));
    x += identity;
    return x.relu_();
}

rcnn::layers::Conv2d Conv3x3(int64_t in_planes, int64_t out_planes, int64_t stride, int64_t groups){
  return rcnn::layers::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 3)
                                        .stride(stride)
                                        .padding(1)
                                        .groups(groups)
                                        .with_bias(false));
}

rcnn::layers::Conv2d Conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride){
  return rcnn::layers::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 1)
                                        .stride(stride)
                                        .with_bias(false));
}

std::map<std::string, ResNetImpl::StageSpec> ResBackbonesMap(){
  ResNetImpl::StageSpec R50C4("Bottleneck", {3, 4, 6, 3}, 4, false, 1, 64, rcnn::config::GetCFG<int>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));
  ResNetImpl::StageSpec R50C5("Bottleneck", {3, 4, 6, 3}, 5, false, 1, 64, rcnn::config::GetCFG<int>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));
  ResNetImpl::StageSpec R101C4("Bottleneck", {3, 4, 23, 3}, 4, false, 1, 64, rcnn::config::GetCFG<int>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));
  ResNetImpl::StageSpec R101C5("Bottleneck", {3, 4, 23, 3}, 5, false, 1, 64, rcnn::config::GetCFG<int>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));
  ResNetImpl::StageSpec R50FPN("Bottleneck", {3, 4, 6, 3}, 5, true, 1, 64, rcnn::config::GetCFG<int>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));
  ResNetImpl::StageSpec R101FPN("Bottleneck", {3, 4, 23, 3}, 5, true, 1, 64, rcnn::config::GetCFG<int>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));
  ResNetImpl::StageSpec R152FPN("Bottleneck", {3, 8, 36, 3}, 5, true, 1, 64, rcnn::config::GetCFG<int>({"MODEL", "BACKBONE", "FREEZE_CONV_BODY_AT"}));

  std::map<std::string, ResNetImpl::StageSpec> blockMap{
    {"R-50-C4", R50C4},
    {"R-50-C5", R50C5},
    {"R-101-C4", R101C4},
    {"R-101-C5", R101C5},
    {"R-50-FPN", R50FPN},
    {"R-101-FPN", R101FPN},
    {"R-152-FPN", R152FPN},
  };
  return blockMap;
}
}
}