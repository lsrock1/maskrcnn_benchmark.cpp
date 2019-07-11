#include "detector/generalized_rcnn.h"
#include <iostream>


namespace rcnn{
namespace modeling{

GeneralizedRCNNImpl::GeneralizedRCNNImpl() 
  :backbone(register_module("backbone", BuildBackbone())), 
   rpn(register_module("rpn", BuildRPN(backbone->get_out_channels()))),
   roi_heads(register_module("roi_heads", BuildROIHeads(backbone->get_out_channels()))){}

std::shared_ptr<GeneralizedRCNNImpl> GeneralizedRCNNImpl::clone(torch::optional<torch::Device> device) const{
  torch::NoGradGuard no_grad;
  std::shared_ptr<GeneralizedRCNNImpl> copy = std::make_shared<GeneralizedRCNNImpl>();
  auto named_params = named_parameters();
  auto named_bufs = named_buffers();
  for(auto& i : copy->named_parameters()){
    i.value().copy_(named_params[i.key()]);
  }
  for(auto& i : copy->named_buffers()){
    i.value().copy_(named_bufs[i.key()]);
  }
  if(device.has_value())
    copy->to(device.value());
  return copy;
}

std::vector<rcnn::structures::BoxList> GeneralizedRCNNImpl::forward(std::vector<torch::Tensor> images){
  assert(!is_training());
  std::vector<rcnn::structures::BoxList> result, proposals;
  torch::Tensor x;
  rcnn::structures::ImageList imageList = rcnn::structures::ToImageList(images);
  std::map<std::string, torch::Tensor> proposal_losses, detector_losses, losses;
  
  std::vector<torch::Tensor> features = backbone->forward(imageList.get_tensors());
  std::tie(proposals, proposal_losses) = rpn->forward(imageList, features);
  if(roi_heads)
    std::tie(x, result, detector_losses) = roi_heads->forward(features, proposals);
  else
    result = proposals;

  return result;
};

std::vector<rcnn::structures::BoxList> GeneralizedRCNNImpl::forward(rcnn::structures::ImageList images){
  assert(!is_training());
  std::vector<rcnn::structures::BoxList> result, proposals;
  torch::Tensor x;
  rcnn::structures::ImageList imageList = rcnn::structures::ToImageList(images);
  std::map<std::string, torch::Tensor> proposal_losses, detector_losses, losses;
  
  std::vector<torch::Tensor> features = backbone->forward(imageList.get_tensors());
  std::tie(proposals, proposal_losses) = rpn->forward(imageList, features);
  
  if(roi_heads)
    std::tie(x, result, detector_losses) = roi_heads->forward(features, proposals);
  else
    result = proposals;

  return result;
};

template<>
std::vector<rcnn::structures::BoxList> GeneralizedRCNNImpl::forward<std::vector<rcnn::structures::BoxList>>(rcnn::structures::ImageList images, std::vector<rcnn::structures::BoxList> targets){
  assert(!is_training());
  std::vector<rcnn::structures::BoxList> result, proposals;
  torch::Tensor x;
  rcnn::structures::ImageList imageList = rcnn::structures::ToImageList(images);
  std::map<std::string, torch::Tensor> proposal_losses, detector_losses, losses;

  std::vector<torch::Tensor> features = backbone->forward(imageList.get_tensors());
  std::tie(proposals, proposal_losses) = rpn->forward(imageList, features, targets);

  if(roi_heads)
    std::tie(x, result, detector_losses) = roi_heads->forward(features, proposals, targets);
  else
    result = proposals;

  return result;
};

template<>
std::map<std::string, torch::Tensor> GeneralizedRCNNImpl::forward<std::map<std::string, torch::Tensor>>(std::vector<torch::Tensor> images, std::vector<rcnn::structures::BoxList> targets){
  assert(is_training());
  std::vector<rcnn::structures::BoxList> result, proposals;
  torch::Tensor x;
  rcnn::structures::ImageList imageList = rcnn::structures::ToImageList(images);
  std::map<std::string, torch::Tensor> proposal_losses, detector_losses, losses;

  std::vector<torch::Tensor> features = backbone->forward(imageList.get_tensors());
  std::tie(proposals, proposal_losses) = rpn->forward(imageList, features, targets);

  if(roi_heads)
    std::tie(x, result, detector_losses) = roi_heads->forward(features, proposals, targets);
  else
    result = proposals;

  losses.insert(detector_losses.begin(), detector_losses.end());
  losses.insert(proposal_losses.begin(), proposal_losses.end());
  return losses;
};

template<>
std::map<std::string, torch::Tensor> GeneralizedRCNNImpl::forward<std::map<std::string, torch::Tensor>>(rcnn::structures::ImageList images, std::vector<rcnn::structures::BoxList> targets){
  assert(is_training());
  std::vector<rcnn::structures::BoxList> result, proposals;
  torch::Tensor x;
  rcnn::structures::ImageList imageList = rcnn::structures::ToImageList(images);
  std::map<std::string, torch::Tensor> proposal_losses, detector_losses, losses;

  std::vector<torch::Tensor> features = backbone->forward(imageList.get_tensors());
  std::tie(proposals, proposal_losses) = rpn->forward(imageList, features, targets);
  if(roi_heads)
    std::tie(x, result, detector_losses) = roi_heads->forward(features, proposals, targets);
  else
    result = proposals;

  losses.insert(detector_losses.begin(), detector_losses.end());
  losses.insert(proposal_losses.begin(), proposal_losses.end());
  return losses;
}

}
}