#pragma once
#include "conv2d.h"
#include "batch_norm.h"


namespace rcnn{
namespace modeling{

class BottleneckImpl : public torch::nn::Module{
  public:
      BottleneckImpl(int64_t in_channels, int64_t bottleneck_channels, int64_t out_channels, int64_t num_groups, bool stride_in_1x1, int64_t stride, int64_t dilation/*, norm_func , dcn_config*/);
      torch::Tensor forward(torch::Tensor x);
  
  private:
    rcnn::layers::Conv2d conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr};
    rcnn::layers::FrozenBatchNorm2d bn1_{nullptr}, bn2_{nullptr}, bn3_{nullptr};
    torch::nn::Sequential downsample_{nullptr};
};

TORCH_MODULE(Bottleneck);

class BaseStemImpl : public torch::nn::Module{
  public:
    BaseStemImpl();
    torch::Tensor forward(torch::Tensor& x);

  private:
    rcnn::layers::Conv2d conv1_{nullptr};
    rcnn::layers::FrozenBatchNorm2d bn1_{nullptr};
};

TORCH_MODULE(BaseStem);

class ResNetImpl : public torch::nn::Module{
  public:
    struct StageSpec{
      StageSpec(int index, int block_count, bool return_features);
      int index_;
      int block_count_;
      bool return_features_;
    };

    ResNetImpl();
    std::vector<torch::Tensor> forward(torch::Tensor x);

  private:
    void freeze_backbone(int freeze_at);
    BaseStem stem_{nullptr};
    std::vector<torch::nn::Sequential> stages_;
    std::vector<bool> return_features_;
};

TORCH_MODULE(ResNet);

class ResNetHeadImpl : public torch::nn::Module{
  public:
    ResNetHeadImpl(/*std::string block_module, */std::vector<ResNetImpl::StageSpec> stages, int64_t num_groups=1, int64_t width_per_groups=64, bool stride_in_1x1=true, int64_t stride_init=0, int64_t res2_out_channels=256, int64_t dilation=1);
    torch::Tensor forward(torch::Tensor x);
    int64_t out_channels() const;

  private:
    int64_t out_channels_;
    std::vector<torch::nn::Sequential> stages_;
};

TORCH_MODULE(ResNetHead);

torch::nn::Sequential MakeStage(/*transformation_module, */int64_t in_channels, int64_t bottleneck_channels, int64_t out_channels, int64_t block_count, int64_t num_groups, bool stride_in_1x1, int64_t first_stride, int64_t dilation=1);

}//namespace modeling

namespace registry{

// const _STEM_MODULES
std::vector<rcnn::modeling::ResNetImpl::StageSpec> STAGE_SPECS(std::string name);
}

}//namespace rcnn

