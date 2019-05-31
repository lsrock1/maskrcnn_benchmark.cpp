#pragma once
#include <torch/torch.h>
#include "backbone/resnet.h"
#include "bounding_box.h"
#include "poolers.h"


namespace rcnn{
namespace modeling{

class ResNet50Conv5ROIFeatureExtractorImpl : public torch::nn::Module{
  public:
    ResNet50Conv5ROIFeatureExtractorImpl(int64_t in_channels);
    torch::Tensor forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> proposals);
    int64_t out_channels() const;

  private:
    Pooler pooler_;
    ResNetHead head_;
    int64_t out_channels_;
};

TORCH_MODULE(ResNet50Conv5ROIFeatureExtractor);

class FPN2MLPFeatureExtractorImpl : public torch::nn::Module{
  public:
    FPN2MLPFeatureExtractorImpl(int64_t in_channels);
    torch::Tensor forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> proposals);
    int64_t out_channels() const;

  private:
    Pooler pooler_;
    torch::nn::Linear fc6_;
    torch::nn::Linear fc7_;
    int64_t out_channels_;
};

TORCH_MODULE(FPN2MLPFeatureExtractor);

class FPNXconv1fcFeatureExtractorImpl : public torch::nn::Module{
  public:
    FPNXconv1fcFeatureExtractorImpl(int64_t in_channels);
    torch::Tensor forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> proposals);
    int64_t out_channels () const;

  private:
    torch::nn::Sequential make_xconvs(int64_t in_channels);
    Pooler pooler_;
    torch::nn::Sequential xconvs_;
    torch::nn::Linear fc6_;
    int64_t out_channels_;
};

TORCH_MODULE(FPNXconv1fcFeatureExtractor);

torch::nn::Sequential MakeROIBoxFeatureExtractor(int64_t in_channels);

}
}