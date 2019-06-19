#pragma once
#include <torch/torch.h>
#include "matcher.h"
#include "bounding_box.h"
#include "segmentation_mask.h"
#include "defaults.h"


namespace rcnn{
namespace modeling{

torch::Tensor ProjectMasksOnBoxes(torch::Tensor segmentation_masks, rcnn::structures::BoxList proposals, int discretization_size);

class MaskRCNNLossComputation{

public:
  MaskRCNNLossComputation(Matcher* proposal_matcher, int discretization_size);
  ~MaskRCNNLossComputation();
  MaskRCNNLossComputation(const MaskRCNNLossComputation& other);
  MaskRCNNLossComputation(MaskRCNNLossComputation&& other);
  MaskRCNNLossComputation operator=(const MaskRCNNLossComputation& other);
  MaskRCNNLossComputation operator=(MaskRCNNLossComputation&& other);
  std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PrepareTargets(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
  rcnn::structures::BoxList MatchTargetsToProposals(rcnn::structures::BoxList proposal, rcnn::structures::BoxList target);
  torch::Tensor operator()(std::vector<rcnn::structures::BoxList> proposals, torch::Tensor mask_logits, std::vector<rcnn::structures::BoxList> targets);

private:
  Matcher* proposal_matcher_{nullptr};
  int discretization_size_;

};

MaskRCNNLossComputation MakeROIMaskLossEvaluator();

}
}