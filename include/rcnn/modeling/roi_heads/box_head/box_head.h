#pragma once
#include <torch/torch.h>
#include "bounding_box.h"
#include "roi_heads/box_head/inference.h"
#include "roi_heads/box_head/loss.h"
#include "roi_heads/box_head/roi_box_feature_extractors.h"
#include "roi_heads/box_head/roi_box_predictors.h"


namespace rcnn{
namespace modeling{

namespace{
  using proposals = std::vector<rcnn::structures::BoxList>;
  using losses = std::map<std::string, torch::Tensor>;
}

class ROIBoxHeadImpl : public torch::nn::Module{
  public:
    ROIBoxHeadImpl(int64_t in_channels);
    std::tuple<torch::Tensor, proposals, losses> forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
    std::tuple<torch::Tensor, proposals, losses> forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals);

  private:
    torch::nn::Sequential feature_extractor_;
    torch::nn::Sequential predictor_;
    PostProcessor post_processor_;
    FastRCNNLossComputation loss_evaluator_;
};

TORCH_MODULE(ROIBoxHead);

ROIBoxHead BuildROIBoxHead(int64_t in_channels);

}
}