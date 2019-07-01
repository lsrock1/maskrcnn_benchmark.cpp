#include "roi_heads/roi_heads.h"


namespace rcnn{
namespace modeling{

CombinedROIHeadsImpl::CombinedROIHeadsImpl(std::set<std::string> heads, int64_t in_channels){
  if(heads.count("mask"))
    mask = register_module("mask", BuildROIMaskHead(in_channels));
  if(heads.count("box"))
    box = register_module("box", BuildROIBoxHead(in_channels));
}

std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> CombinedROIHeadsImpl::forward(std::vector<torch::Tensor> features, 
                                                                                                                std::vector<rcnn::structures::BoxList> proposals, 
                                                                                                                std::vector<rcnn::structures::BoxList> targets)
{
  std::map<std::string, torch::Tensor> losses;
  torch::Tensor x;
  std::vector<rcnn::structures::BoxList> detections;
  std::map<std::string, torch::Tensor> loss_box;
  
  std::tie(x, detections, loss_box) = box->forward(features, proposals, targets);
  losses.insert(loss_box.begin(), loss_box.end());

  if(mask){
    std::map<std::string, torch::Tensor> loss_mask;

    if(is_training() && rcnn::config::GetCFG<bool>({"MODEL", "ROI_MASK_HEAD", "SHARE_BOX_FEATURE_EXTRACTOR"}))
      std::tie(x, detections, loss_mask) = mask->forward(x, detections, targets);
    else
      std::tie(x, detections, loss_mask) = mask->forward(features, detections, targets);

    losses.insert(loss_mask.begin(), loss_mask.end());
  }

  return std::make_tuple(x, detections, losses);
}

std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> CombinedROIHeadsImpl::forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals){
  std::map<std::string, torch::Tensor> losses;
  torch::Tensor x;
  std::vector<rcnn::structures::BoxList> detections;
  std::map<std::string, torch::Tensor> loss_box;
  
  std::tie(x, detections, loss_box) = box->forward(features, proposals);
  losses.insert(loss_box.begin(), loss_box.end());

  if(mask){
    std::map<std::string, torch::Tensor> loss_mask;
    std::tie(x, detections, loss_mask) = mask->forward(features, detections);
    losses.insert(loss_mask.begin(), loss_mask.end());
  }

  return std::make_tuple(x, detections, losses);
}


CombinedROIHeads BuildROIHeads(int64_t out_channels){
  std::set<std::string> roi_heads;
  if(!rcnn::config::GetCFG<bool>({"MODEL", "RPN_ONLY"})){
    roi_heads.insert("box");
  }
  else{
    return nullptr;
  }
  if(rcnn::config::GetCFG<bool>({"MODEL", "MASK_ON"}))
    roi_heads.insert("mask");
  return CombinedROIHeads(roi_heads, out_channels);
}

}
}