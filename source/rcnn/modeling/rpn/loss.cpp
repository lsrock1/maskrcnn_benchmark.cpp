#include "rpn/loss.h"
#include "smooth_l1_loss.h"
#include "defaults.h"
#include "rpn/utils.h"
#include <cassert>


namespace rcnn{
namespace modeling{


RPNLossComputation::RPNLossComputation(Matcher proposal_matcher, BalancedPositiveNegativeSampler fg_bg_sampler, 
                                        BoxCoder box_coder, LabelGenerater generate_labels_func)
                                    : proposal_matcher_(proposal_matcher),
                                        fg_bg_sampler_(fg_bg_sampler),
                                        box_coder_(box_coder),
                                        generate_labels_func_(generate_labels_func){}

rcnn::structures::BoxList RPNLossComputation::MatchTargetsToAnchors(rcnn::structures::BoxList& anchor, rcnn::structures::BoxList& target, const std::vector<std::string> copied_fields){
  torch::Tensor match_quality_matrix = rcnn::structures::BoxList::BoxListIOU(target, anchor);
  torch::Tensor matched_idxs = proposal_matcher_(match_quality_matrix);

  target = target.CopyWithFields(copied_fields);
  rcnn::structures::BoxList matched_targets = target[matched_idxs.clamp(/*min=*/0)];
  matched_targets.AddField("matched_idxs", matched_idxs);
  return matched_targets;
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> RPNLossComputation::PrepareTargets(std::vector<rcnn::structures::BoxList>& anchors, std::vector<rcnn::structures::BoxList>& targets){
  std::vector<torch::Tensor> labels;
  std::vector<torch::Tensor> regression_targets;
  assert(anchors.size() == targets.size());
  for(int i = 0; i < anchors.size(); ++i){
    rcnn::structures::BoxList matched_targets = MatchTargetsToAnchors(anchors[i], targets[i], copied_fields_);
    torch::Tensor matched_idxs = matched_targets.GetField("matched_idxs");
    torch::Tensor labels_per_image = generate_labels_func_(matched_targets);
    labels_per_image = labels_per_image.to(torch::kF32);

    torch::Tensor bg_indices = (matched_idxs == Matcher::BELOW_LOW_THRESHOLD);
    labels_per_image.masked_fill(bg_indices, 0);

    if(discard_cases_.count("not_visibility") > 0)
      labels_per_image.masked_fill(1 - anchors[i].GetField("visibility"), -1);

    if(discard_cases_.count("between_thresholds") > 0){
      torch::Tensor inds_to_discard = (matched_idxs == Matcher::BETWEEN_THRESHOLDS);
      labels_per_image.masked_fill(inds_to_discard, -1);
    }

    labels.push_back(labels_per_image);
    regression_targets.push_back(box_coder_.encode(
      matched_targets.get_bbox(), anchors[i].get_bbox()
    ));
  }
  return std::make_pair(labels, regression_targets);
}

std::pair<torch::Tensor, torch::Tensor> RPNLossComputation::operator() (std::vector<std::vector<rcnn::structures::BoxList>>& anchors, std::vector<torch::Tensor>& objectness, std::vector<torch::Tensor>& box_regression, std::vector<rcnn::structures::BoxList>& targets){
  std::vector<rcnn::structures::BoxList> cat_boxlists;
  std::vector<torch::Tensor> labels, regression_targets, sampled_pos_inds, sampled_neg_inds;
  torch::Tensor objectness_tensor, box_regression_tensor, labels_tensor, regression_targets_tensor, sampled_inds, sampled_pos_inds_tensor, sampled_neg_inds_tensor;

  for(auto& anchors_per_image: anchors)
    cat_boxlists.push_back(rcnn::structures::BoxList::CatBoxList(anchors_per_image));

  std::tie(labels, regression_targets) = PrepareTargets(cat_boxlists, targets);
  std::tie(sampled_pos_inds, sampled_neg_inds) = fg_bg_sampler_(labels);
  sampled_pos_inds_tensor = torch::nonzero(torch::cat(sampled_pos_inds, 0)).squeeze(1);
  sampled_neg_inds_tensor = torch::nonzero(torch::cat(sampled_neg_inds, 0)).squeeze(1);

  sampled_inds = torch::cat({sampled_pos_inds_tensor, sampled_neg_inds_tensor}, /*dim=*/0);
  std::tie(objectness_tensor, box_regression_tensor) = ConcatBoxPredictionLayers(objectness, box_regression);
  objectness_tensor.squeeze_();
  labels_tensor = torch::cat(labels, 0);
  regression_targets_tensor = torch::cat(regression_targets, 0);

  return std::make_pair(
    torch::binary_cross_entropy_with_logits(objectness_tensor.index_select(0, sampled_inds), labels_tensor.index_select(0, sampled_inds), {}, {}, Reduction::Mean),
    rcnn::layers::smooth_l1_loss(box_regression_tensor.index_select(0, sampled_pos_inds_tensor), regression_targets_tensor.index_select(0, sampled_pos_inds_tensor), 1./9., false)
  );
}

torch::Tensor GenerateRPNLabels(rcnn::structures::BoxList matched_targets){
  torch::Tensor matched_idxs = matched_targets.GetField("matched_idxs");
  torch::Tensor labels_per_image = matched_idxs >= 0;
  return labels_per_image;
}

RPNLossComputation MakeRPNLossEvaluator(BoxCoder box_coder){
  Matcher matcher = Matcher(
    rcnn::config::GetCFG<float>({"MODEL", "RPN", "FG_IOU_THRESHOLD"}),
    rcnn::config::GetCFG<float>({"MODEL", "RPN", "BG_IOU_THRESHOLD"}),
    /*allow_low_quality_matches=*/true
  );

  BalancedPositiveNegativeSampler fg_bg_sampler_a = rcnn::modeling::BalancedPositiveNegativeSampler(
    rcnn::config::GetCFG<int64_t>({"MODEL", "RPN", "BATCH_SIZE_PER_IMAGE"}),
    rcnn::config::GetCFG<float>({"MODEL", "RPN", "POSITIVE_FRACTION"})
  );

  return RPNLossComputation(
    matcher,
    fg_bg_sampler_a,
    box_coder,
    GenerateRPNLabels
  );
}

}
}