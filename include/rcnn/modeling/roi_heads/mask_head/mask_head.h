#pragma once
#include <torch/torch.h>

#include <bounding_box.h>

#include "roi_heads/mask_head/roi_mask_predictors.h"
#include "roi_heads/mask_head/roi_mask_feature_extractors.h"
#include "roi_heads/mask_head/loss.h"
#include "roi_heads/mask_head/inference.h"


namespace rcnn{
namespace modeling{

std::pair<std::vector<rcnn::structures::BoxList>, std::vector<torch::Tensor>> KeepOnlyPositiveBoxes(std::vector<rcnn::structures::BoxList> boxes);

class ROIMaskHeadImpl : public torch::nn::Module{

public:
  ROIMaskHeadImpl(int64_t in_channels);
  std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
  std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals);
  std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(torch::Tensor features, std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);

private:
  MaskPostProcessor post_processor;
  MaskRCNNLossComputation loss_evaluator;
  MaskRCNNFPNFeatureExtractor feature_extractor;
  torch::nn::Sequential predictor;
};

TORCH_MODULE(ROIMaskHead);

ROIMaskHead BuildROIMaskHead(int64_t in_channels);

}
}