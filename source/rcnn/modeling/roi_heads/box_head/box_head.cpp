#include "roi_heads/box_head/box_head.h"


namespace rcnn{
namespace modeling{

ROIBoxHeadImpl::ROIBoxHeadImpl(int64_t in_channels)
                              :feature_extractor_(register_module("feature_extractor", MakeROIBoxFeatureExtractor(in_channels))),
                               predictor_(register_module("predictor", MakeROIBoxPredictor(in_channels))),
                               post_processor_(MakeROIBoxPostProcessor()),
                               loss_evaluator_(MakeROIBoxLossEvaluator()){}

std::tuple<torch::Tensor, proposals, losses> ROIBoxHeadImpl::forward(std::vector<torch::Tensor> features, 
                                                                     std::vector<rcnn::structures::BoxList> proposals, 
                                                                     std::vector<rcnn::structures::BoxList> targets)
{
  if(is_training()){
    {
      torch::NoGradGuard guard;
      proposals = loss_evaluator_.Subsample(proposals, targets);
    }
  }
  return forward(features, proposals);
}

std::tuple<torch::Tensor, proposals, losses> ROIBoxHeadImpl::forward(std::vector<torch::Tensor> features, 
                                                                     std::vector<rcnn::structures::BoxList> proposals)
{
  torch::Tensor x = feature_extractor_->forward(features, proposals);
  torch::Tensor class_logits, box_regression;
  std::tie(class_logits, box_regression) = predictor_->forward<std::pair<torch::Tensor, torch::Tensor>>(x);

  if(!is_training()){
    std::vector<rcnn::structures::BoxList> result = post_processor_(std::make_pair(class_logits, box_regression), proposals);
    return std::make_tuple(x, result, losses());
  }

  torch::Tensor loss_classifier, loss_box_reg;
  std::tie(loss_classifier, loss_box_reg) = loss_evaluator_(std::vector<torch::Tensor>{class_logits}, std::vector<torch::Tensor>{box_regression});
  return std::make_tuple(
    x, proposals,
    losses{
      {"loss_classifier", loss_classifier},
      {"loss_box_reg", loss_box_reg}
    }
  );
}

ROIBoxHead BuildROIBoxHead(int64_t in_channels){
  return ROIBoxHead(in_channels);
}

}
}