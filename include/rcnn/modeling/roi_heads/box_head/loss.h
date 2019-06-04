#pragma once
#include <torch/torch.h>
#include "matcher.h"
#include "balanced_positive_negative_sampler.h"
#include "box_coder.h"
#include "bounding_box.h"


namespace rcnn{
namespace modeling{

class FastRCNNLossComputation{
  public:
    FastRCNNLossComputation(Matcher proposal_matcher, BalancedPositiveNegativeSampler fg_bg_sampler, BoxCoder box_coder, bool cls_agnostic_bbox_reg=false);
    rcnn::structures::BoxList MatchTargetsToProposals(rcnn::structures::BoxList proposal, rcnn::structures::BoxList target);
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PrepareTargets(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
    std::vector<rcnn::structures::BoxList> Subsample(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
    std::pair<torch::Tensor, torch::Tensor> operator()(std::vector<torch::Tensor> class_logits, std::vector<torch::Tensor> box_regression);

  private:
    std::vector<rcnn::structures::BoxList> _proposals;
    Matcher proposal_matcher_;
    BalancedPositiveNegativeSampler fg_bg_sampler_;
    BoxCoder box_coder_;
    bool cls_agnostic_bbox_reg_;
};

FastRCNNLossComputation MakeROIBoxLossEvaluator();

}
}