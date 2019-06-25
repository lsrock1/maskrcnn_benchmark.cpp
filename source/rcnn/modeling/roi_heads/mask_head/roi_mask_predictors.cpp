#include "roi_heads/mask_head/roi_mask_predictors.h"
#include "defaults.h"
#include <cassert>


namespace rcnn{
namespace modeling{

MaskRCNNC4PredictorImpl::MaskRCNNC4PredictorImpl(int64_t in_channels)
                                                :conv5_mask(
                                                   register_module("conv5_mask",
                                                     rcnn::layers::Conv2d(
                                                       torch::nn::Conv2dOptions(
                                                         in_channels, 
                                                         rcnn::config::GetCFG<std::vector<int64_t>>({"MODEL", "ROI_MASK_HEAD", "CONV_LAYERS"}).back(),
                                                         2//kernel_size
                                                       ).stride(2).padding(0).transposed(true)
                                                     )
                                                   )//register_module
                                                 ),//conv5_mask
                                                 mask_fcn_logits(
                                                   register_module("mask_fcn_logits",
                                                     rcnn::layers::Conv2d(
                                                       torch::nn::Conv2dOptions(
                                                         rcnn::config::GetCFG<std::vector<int64_t>>({"MODEL", "ROI_MASK_HEAD", "CONV_LAYERS"}).back(), 
                                                         rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "NUM_CLASSES"}),
                                                         1//kernel_size
                                                       ).stride(1).padding(0)
                                                     )
                                                   )
                                                 )
{
  torch::nn::init::kaiming_normal_(conv5_mask->weight, 0, torch::nn::init::FanMode::FanOut, torch::nn::init::Nonlinearity::ReLU);
  torch::nn::init::constant_(conv5_mask->bias, 0);
  torch::nn::init::kaiming_normal_(mask_fcn_logits->weight, 0, torch::nn::init::FanMode::FanOut, torch::nn::init::Nonlinearity::ReLU);
  torch::nn::init::constant_(mask_fcn_logits->bias, 0);
}

torch::Tensor MaskRCNNC4PredictorImpl::forward(torch::Tensor x){
  x = conv5_mask->forward(x).relu_();
  return mask_fcn_logits->forward(x);
}

MaskRCNNConv1x1PredictorImpl::MaskRCNNConv1x1PredictorImpl(int64_t in_channels)
                                                          :mask_fcn_logits(
                                                             register_module("mask_fcn_logits",
                                                               rcnn::layers::Conv2d(
                                                                 torch::nn::Conv2dOptions(
                                                                   in_channels, 
                                                                   rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "NUM_CLASSES"}),
                                                                   1//kernel_size
                                                                 ).stride(1).padding(0)
                                                               )
                                                             )
                                                           )
{
  torch::nn::init::kaiming_normal_(mask_fcn_logits->weight, 0, torch::nn::init::FanMode::FanOut, torch::nn::init::Nonlinearity::ReLU);
  torch::nn::init::constant_(mask_fcn_logits->bias, 0);
}

torch::Tensor MaskRCNNConv1x1PredictorImpl::forward(torch::Tensor x){
  return mask_fcn_logits->forward(x);
}

torch::nn::Sequential MakeROIMaskPredictor(int64_t in_channels){
  std::string name = rcnn::config::GetCFG<std::string>({"MODEL", "ROI_MASK_HEAD", "PREDICTOR"});
  torch::nn::Sequential predictor;
  if(name.compare("MaskRCNNC4Predictor") == 0){
    predictor->push_back(MaskRCNNC4Predictor(in_channels));
  }
  else if(name.compare("MaskRCNNConv1x1Predictor") == 0){
    predictor->push_back(MaskRCNNConv1x1Predictor(in_channels));
  }
  else{
    assert(false);
  }
  return predictor;
}

}
}