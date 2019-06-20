#include "detector/generalized_rcnn.h"


namespace rcnn{
namespace modeling{

GeneralizedRCNNImpl::GeneralizedRCNNImpl() : backbone(BuildBackbone()), rpn(BuildRPN(backbone->get_out_channels())), roi_heads(BuildROIBoxHead(backbone->get_out_channels())){}

template<>
std::map<std::string, torch::Tensor> GeneralizedRCNNImpl::forward(std::vector<torch::Tensor> images, std::vector<rcnn::structures::BoxList> targets){
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

template<>
std::vector<rcnn::structures::BoxList> GeneralizedRCNNImpl::forward(rcnn::structures::ImageList images, std::vector<rcnn::structures::BoxList> targets){
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

}
}