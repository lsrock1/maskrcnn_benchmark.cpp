#pragma once
#include <torch/torch.h>
#include "conv2d.h"
#include "batch_norm.h"
#include <vector>
#include <string>

namespace rcnn{
namespace modeling{
  
  class ResNetImpl : public torch::nn::Module{
    public:
      class StageSpec{
        public:
          StageSpec(std::string block, std::initializer_list<int64_t> num_layers, int stage_to, bool is_fpn, int64_t groups, int64_t width_per_group);
          int get_stage_to();
          std::initializer_list<int64_t> get_num_layers();
          bool get_is_fpn();
          std::string get_block();
          int64_t get_groups();
          int64_t get_width_per_group();
          void set_freeze_at(int at);
          int get_freeze_at();
          // int64_t get_out_channels();
          // int64_t get_res2_out_channels();

        private:
          int freeze_at_;
          std::string block_;
          std::initializer_list<int64_t> num_layers_;
          int stage_to_;
          bool is_fpn_;
          int64_t groups_;
          int64_t width_per_group_;
          // int64_t res2_out_channels_ = 64;
          // int64_t out_channels_ = 512
      };
      ResNetImpl(StageSpec& stage_spec);
      torch::Tensor forward(torch::Tensor x);
      std::vector<torch::Tensor> forward_fpn(torch::Tensor x);
      const static int64_t res2_out_channels = 256;
      const static int64_t out_channels = 256;
      bool get_is_fpn();
      int64_t get_out_channels();
      int64_t get_res2_out_channels();

    private:      
      torch::nn::Sequential MakeLayer(int64_t planes, int64_t blocks, int64_t stride=1);
      void initialize();
      void freeze_backbone(int freeze_at);

      std::string block_;
      bool is_fpn_;
      int64_t in_planes_ = 64;
      int64_t groups_;
      int64_t base_width_;
      int64_t expansion_;
      rcnn::layers::Conv2d conv1_;
      rcnn::layers::FrozenBatchNorm2d bn1_;
      torch::nn::Sequential layer1_{nullptr}, layer2_{nullptr}, layer3_{nullptr}, layer4_{nullptr};
  };

  TORCH_MODULE(ResNet);

  class BasicBlockImpl : public torch::nn::Module{    
    public:
      BasicBlockImpl(int64_t in_planes, int64_t out_planes, torch::nn::Sequential downsample, int64_t stride=1, int64_t groups=1, int64_t base_width=64/*c++ frontend only has batch norm*/);
      BasicBlockImpl(int64_t in_planes, int64_t out_planes, int64_t stride=1, int64_t groups=1, int64_t base_width=64/*c++ frontend only has batch norm*/);
      torch::Tensor forward(torch::Tensor x);
      const static int64_t kExpansion = 1;

    private:
      int64_t width_;
      int64_t stride_;
      rcnn::layers::Conv2d conv1_, conv2_;
      rcnn::layers::FrozenBatchNorm2d bn1_, bn2_;
      torch::nn::Sequential downsample_{nullptr};
  };

  TORCH_MODULE(BasicBlock);

  class BottleneckImpl : public torch::nn::Module{
    public:
        BottleneckImpl(int64_t in_planes, int64_t out_planes, torch::nn::Sequential downsample, int64_t stride=1, int64_t groups=1, int64_t base_width=64);
        BottleneckImpl(int64_t in_planes, int64_t out_planes, int64_t stride=1, int64_t groups=1, int64_t base_width=64/*c++ frontend only has batch norm*/);
        torch::Tensor forward(torch::Tensor x);
        const static int64_t kExpansion = 4;
    
    private:
      int64_t stride_;
      int64_t width_;
      rcnn::layers::Conv2d conv1_, conv2_, conv3_;
      rcnn::layers::FrozenBatchNorm2d bn1_, bn2_, bn3_;
      torch::nn::Sequential downsample_{nullptr};
  };

  TORCH_MODULE(Bottleneck);

  rcnn::layers::Conv2d Conv3x3(int64_t in_planes, int64_t out_planes, int64_t stride=1, int64_t groups=1);
  rcnn::layers::Conv2d Conv1x1(int64_t in_planes, int64_t out_planes, int64_t stride=1);
  std::map<std::string, ResNetImpl::StageSpec> ResBackbones();

}//namespace resnet
}//namespace model