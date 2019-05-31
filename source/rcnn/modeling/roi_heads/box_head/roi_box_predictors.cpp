#include "roi_heads/box_head/roi_box_predictors.h"
#include "defaults.h"
#include <cassert>


namespace rcnn{
namespace modeling{

FastRCNNPredictorImpl::FastRCNNPredictorImpl(int64_t in_channels)
  :cls_score_(
    register_module("cls_score",
      torch::nn::Linear(in_channels, rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "NUM_CLASSES"}))
    )//register_module
  ),//cls_score
  bbox_pred_(
    register_module("bbox_pred",
      torch::nn::Linear(
        in_channels,
        (rcnn::config::GetCFG<bool>({"MODEL", "CLS_AGNOSTIC_BBOX_REG"}) ? 2 : rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "NUM_CLASSES"})) * 4
      )
  ))
{
  torch::nn::init::normal_(cls_score_->weight, 0, 0.01);
  torch::nn::init::constant_(cls_score_->bias, 0);
  torch::nn::init::normal_(bbox_pred_->weight, 0, 0.001);
  torch::nn::init::constant_(bbox_pred_->bias, 0);
}

std::vector<torch::Tensor> FastRCNNPredictorImpl::forward(torch::Tensor x){
  x = torch::adaptive_avg_pool2d(x, {1});
  x = x.reshape({x.size(0), -1});
  //cls_logit, bbox_pred
  return std::vector<torch::Tensor>{cls_score_(x), bbox_pred_(x)};
}

FPNPredictorImpl::FPNPredictorImpl(int64_t in_channels)
  :cls_score_(
    register_module("cls_score",
      torch::nn::Linear(in_channels, rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "NUM_CLASSES"}))
    )//register_module
  ),//cls_score
  bbox_pred_(
    register_module("bbox_pred",
      torch::nn::Linear(
        in_channels,
        (rcnn::config::GetCFG<bool>({"MODEL", "CLS_AGNOSTIC_BBOX_REG"}) ? 2 : rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "NUM_CLASSES"})) * 4
      )
  ))
{
  torch::nn::init::normal_(cls_score_->weight, 0, 0.01);
  torch::nn::init::constant_(cls_score_->bias, 0);
  torch::nn::init::normal_(bbox_pred_->weight, 0, 0.001);
  torch::nn::init::constant_(bbox_pred_->bias, 0);
}

std::vector<torch::Tensor> FPNPredictorImpl::forward(torch::Tensor x){
  if(x.ndimension() == 4){
    assert(x.size(2) == 1 && x.size(3) == 1);
    x = x.reshape({x.size(0), -1});
  }
  //scores, bbox_deltas
  return std::vector<torch::Tensor>{cls_score_->forward(x), bbox_pred_->forward(x)};
}

torch::nn::Sequential MakeROIBoxPredictor(int64_t in_channels){
  rcnn::config::CFGS name = rcnn::config::GetCFG<rcnn::config::CFGS>({"MODEL", "ROI_BOX_HEAD", "PREDICTOR"});
  std::string predictor_name = name.get();
  torch::nn::Sequential predictor;
  if(predictor_name.compare("FastRCNNPredictor") == 0){
    predictor->push_back(FastRCNNPredictor(in_channels));
  }
  else if(predictor_name.compare("FPNPredictor") == 0){
    predictor->push_back(FPNPredictor(in_channels));
  }
  else{
    assert(false);
  }
  return predictor;
}

}
}