#pragma once
#include <torch/torch.h>
#include "matcher.h"
#include "balanced_positive_negative_sampler.h"
#include "box_coder.h"
#include "bounding_box.h"
#include <set>


namespace rcnn{
namespace modeling{

class RPNLossComputation{
  using LabelGenerater = torch::Tensor (*)(rcnn::structures::BoxList);

  public:
    RPNLossComputation(Matcher proposal_matcher, BalancedPositiveNegativeSampler fg_bg_sampler, BoxCoder box_coder, LabelGenerater generate_labels_func);
    rcnn::structures::BoxList MatchTargetsToAnchors(rcnn::structures::BoxList& anchor, rcnn::structures::BoxList& target, const std::vector<std::string> copied_fields);
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PrepareTargets(std::vector<rcnn::structures::BoxList>& anchors, std::vector<rcnn::structures::BoxList>& targets);
    std::pair<torch::Tensor, torch::Tensor> operator() (std::vector<std::vector<rcnn::structures::BoxList>>& anchors, std::vector<torch::Tensor>& objectness, std::vector<torch::Tensor>& box_regression, std::vector<rcnn::structures::BoxList>& targets);

  private:
    Matcher proposal_matcher_;
    BalancedPositiveNegativeSampler fg_bg_sampler_;
    BoxCoder box_coder_;
    std::vector<std::string> copied_fields_;
    LabelGenerater generate_labels_func_;
    const std::set<std::string> discard_cases_{"not_visibility", "between_thresholds"};
};

torch::Tensor GenerateRPNLabels(rcnn::structures::BoxList matched_targets);
RPNLossComputation MakeRPNLossEvaluator(BoxCoder box_coder);
}
}