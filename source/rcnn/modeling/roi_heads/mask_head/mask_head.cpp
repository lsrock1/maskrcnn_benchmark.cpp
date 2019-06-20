#include "roi_heads/mask_head/mask_head.h"


namespace rcnn{
namespace modeling{

std::pair<std::vector<rcnn::structures::BoxList>, std::vector<torch::Tensor>> KeepOnlyPositiveBoxes(std::vector<rcnn::structures::BoxList> boxes){
  std::vector<rcnn::structures::BoxList> positive_boxes;
  std::vector<torch::Tensor> positive_inds;

  int num_boxes = 0;
  for(auto& boxes_per_image : boxes){
    torch::Tensor labels = boxes_per_image.GetField("labels");
    torch::Tensor inds_mask = labels > 0;
    torch::Tensor inds = inds_mask.nonzero().squeeze(1);
    positive_boxes.push_back(boxes_per_image[inds]);
    positive_inds.push_back(inds_mask);
  }
  return std::make_pair(positive_boxes, positive_inds);
}

ROIMaskHeadImpl::ROIMaskHeadImpl(int64_t in_channels) 
               :feature_extractor(MakeROIMaskFeatureExtractor(in_channels)),
                predictor(MakeROIMaskPredictor(feature_extractor->out_channels())),
                post_processor(MakeRoiMaskPostProcessor()),
                loss_evaluator(MakeROIMaskLossEvaluator()){}

std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> ROIMaskHeadImpl::forward(std::vector<torch::Tensor> features, 
                                       std::vector<rcnn::structures::BoxList> proposals, 
                                       std::vector<rcnn::structures::BoxList> targets)
{
  if(is_training()){
    torch::Tensor mask_logits, x, loss_mask;
    std::vector<rcnn::structures::BoxList> positive_boxes, all_proposals;
    std::vector<torch::Tensor> positive_inds;
    all_proposals = proposals;

    std::tie(positive_boxes, positive_inds) = KeepOnlyPositiveBoxes(proposals);
    x = feature_extractor->forward(features, proposals);
    mask_logits = predictor->forward<torch::Tensor>(x);
    loss_mask = loss_evaluator(proposals, mask_logits, targets);
    return std::make_tuple(x, all_proposals, std::map<std::string, torch::Tensor>{{"loss_mask", loss_mask}});
  }
  else{
    return forward(features, proposals);
  }  
}

std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> ROIMaskHeadImpl::forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals){
  //No target, No training
  torch::Tensor x = feature_extractor->forward(features, proposals);
  torch::Tensor mask_logits = predictor->forward(x);

  std::vector<rcnn::structures::BoxList> result = post_processor(mask_logits, proposals);
  return std::make_tuple(x, result, std::map<std::string, torch::Tensor>{});
}

std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> ROIMaskHeadImpl::forward(torch::Tensor features, std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets){
  //share feature => training
  std::vector<rcnn::structures::BoxList> positive_boxes, all_proposals;
  std::vector<torch::Tensor> positive_inds;
  torch::Tensor mask_logits, x, loss_mask;
  
  all_proposals = proposals;
  std::tie(positive_boxes, positive_inds) = KeepOnlyPositiveBoxes(proposals);
  x = features[torch::cat(positive_inds, 0)];
  mask_logits = predictor->forward(x);
  loss_mask = loss_evaluator(proposals, mask_logits, targets);

  return std::make_tuple(x, all_proposals, std::map<std::string, torch::Tensor>{{"loss_mask", loss_mask}});
}

ROIMaskHead BuildROIMaskHead(int64_t in_channels){
  return ROIMaskHead(in_channels);
}

}
}