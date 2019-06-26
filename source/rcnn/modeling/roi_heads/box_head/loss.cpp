#include "roi_heads/box_head/loss.h"
#include "cat.h"
#include "defaults.h"
#include "smooth_l1_loss.h"
#include "matcher.h"
#include <cassert>


namespace rcnn{
namespace modeling{

FastRCNNLossComputation::FastRCNNLossComputation(Matcher proposal_matcher, 
                                                 BalancedPositiveNegativeSampler fg_bg_sampler, 
                                                 BoxCoder box_coder, 
                                                 bool cls_agnostic_bbox_reg)
                                                 :proposal_matcher_(proposal_matcher),
                                                  fg_bg_sampler_(fg_bg_sampler),
                                                  box_coder_(box_coder),
                                                  cls_agnostic_bbox_reg_(cls_agnostic_bbox_reg){}

rcnn::structures::BoxList FastRCNNLossComputation::MatchTargetsToProposals(rcnn::structures::BoxList proposal, rcnn::structures::BoxList target){
  torch::Tensor match_quality_matrix = rcnn::structures::BoxList::BoxListIOU(target, proposal);
  torch::Tensor matched_idxs = proposal_matcher_(match_quality_matrix);
  target = target.CopyWithFields(std::vector<std::string>{"labels"});
  rcnn::structures::BoxList matched_targets = target[matched_idxs.clamp(0)];
  matched_targets.AddField("matched_idxs", matched_idxs);
  return matched_targets;
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> FastRCNNLossComputation::PrepareTargets(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets){
  std::vector<torch::Tensor> labels;
  std::vector<torch::Tensor> regression_targets;
  for(size_t i = 0; i < proposals.size(); ++i){
    rcnn::structures::BoxList matched_targets = MatchTargetsToProposals(proposals[i], targets[i]);
    torch::Tensor matched_idxs = matched_targets.GetField("matched_idxs");
    torch::Tensor labels_per_image = matched_targets.GetField("labels");
    labels_per_image = labels_per_image.to(torch::kInt64);
    torch::Tensor bg_inds = (matched_idxs == Matcher::BELOW_LOW_THRESHOLD);
    labels_per_image.masked_fill(bg_inds, 0);
    torch::Tensor ignore_inds = (matched_idxs == Matcher::BETWEEN_THRESHOLDS);
    labels_per_image.masked_fill(ignore_inds, -1);
    torch::Tensor regression_targets_per_image = box_coder_.encode(matched_targets.get_bbox(), proposals[i].get_bbox());
    
    labels.push_back(labels_per_image);
    regression_targets.push_back(regression_targets_per_image);
  }
  return std::make_pair(labels, regression_targets);
}

std::vector<rcnn::structures::BoxList> FastRCNNLossComputation::Subsample(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets){
  std::vector<torch::Tensor> labels, regression_targets, sampled_pos_inds, sampled_neg_inds;
  std::tie(labels, regression_targets) = PrepareTargets(proposals, targets);
  std::tie(sampled_pos_inds, sampled_neg_inds) = fg_bg_sampler_(labels);
  assert(labels.size() == regression_targets.size() && regression_targets.size() == proposals.size());

  for(size_t i = 0; i < labels.size(); ++i){
    proposals[i].AddField("labels", labels[i]);
    proposals[i].AddField("regression_targets", regression_targets[i]);
  }

  assert(sampled_pos_inds.size() == sampled_neg_inds.size());
  for(int img_idx = 0; img_idx < sampled_pos_inds.size(); ++img_idx){
    torch::Tensor img_sampled_inds = torch::nonzero(sampled_pos_inds[img_idx].__or__(sampled_neg_inds[img_idx])).squeeze(1);
    rcnn::structures::BoxList proposals_per_image = proposals[img_idx][img_sampled_inds];
    proposals[img_idx] = proposals_per_image;
  }

  _proposals = proposals;
  return proposals;
}

std::pair<torch::Tensor, torch::Tensor> FastRCNNLossComputation::operator()(std::vector<torch::Tensor> class_logits, std::vector<torch::Tensor> box_regression){
  torch::Tensor class_logits_tensor = rcnn::layers::cat(class_logits, 0);
  torch::Tensor box_regression_tensor = rcnn::layers::cat(box_regression, 0);
  auto device = class_logits_tensor.device();

  assert(_proposals.size() > 0);
  std::vector<rcnn::structures::BoxList> proposals = _proposals;
  
  std::vector<torch::Tensor> cat_vec;
  std::for_each(proposals.begin(), proposals.end(), [&cat_vec](rcnn::structures::BoxList proposal){cat_vec.push_back(proposal.GetField("labels"));});
  torch::Tensor labels = rcnn::layers::cat(cat_vec, 0);
  cat_vec.clear();
  std::for_each(proposals.begin(), proposals.end(), [&cat_vec](rcnn::structures::BoxList proposal){cat_vec.push_back(proposal.GetField("regression_targets"));});
  torch::Tensor regression_targets = rcnn::layers::cat(cat_vec, 0);

  torch::Tensor classification_loss = torch::nll_loss(class_logits_tensor.log_softmax(1), labels);

  torch::Tensor sampled_pos_inds_subset = torch::nonzero(labels > 0).squeeze(1);
  torch::Tensor labels_pos = labels.index_select(0, sampled_pos_inds_subset);

  torch::Tensor map_inds;
  if(cls_agnostic_bbox_reg_){
    map_inds = torch::tensor({4, 5, 6, 7}).to(device).to(torch::kI64);
  }
  else{
    map_inds = 4 * labels_pos.unsqueeze(1) + torch::tensor({0, 1, 2, 3}).to(device).to(torch::kI64);
  }
  cat_vec.clear();
  box_regression_tensor = box_regression_tensor.index_select(0, sampled_pos_inds_subset);
  
  for(int i = 0; i < box_regression_tensor.size(0); ++i)
    cat_vec.push_back(box_regression_tensor[i].index_select(0, map_inds[i]));
  box_regression_tensor = torch::stack(cat_vec);
  torch::Tensor box_loss = rcnn::layers::smooth_l1_loss(box_regression_tensor, 
                               regression_targets.index_select(0, sampled_pos_inds_subset),
                               1., false);

  box_loss = box_loss / labels.numel();

  return std::make_pair(classification_loss, box_loss);
}

FastRCNNLossComputation MakeROIBoxLossEvaluator(){
  Matcher matcher = Matcher(
    rcnn::config::GetCFG<float>({"MODEL", "ROI_HEADS", "FG_IOU_THRESHOLD"}),
    rcnn::config::GetCFG<float>({"MODEL", "ROI_HEADS", "BG_IOU_THRESHOLD"}),
    false
  );

  BoxCoder box_coder = BoxCoder(rcnn::config::GetCFG<std::vector<float>>({"MODEL", "ROI_HEADS", "BBOX_REG_WEIGHTS"}));
  BalancedPositiveNegativeSampler fg_bg_sampler = BalancedPositiveNegativeSampler(
    rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_HEADS", "BATCH_SIZE_PER_IMAGE"}),
    rcnn::config::GetCFG<float>({"MODEL", "ROI_HEADS", "POSITIVE_FRACTION"})
  );

  return FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder, rcnn::config::GetCFG<bool>({"MODEL", "CLS_AGNOSTIC_BBOX_REG"}));
}

}
}