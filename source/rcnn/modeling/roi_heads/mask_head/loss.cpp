#include "roi_heads/mask_head/loss.h"
#include <cassert>
#include "cat.h"


namespace rcnn{
namespace modeling{

torch::Tensor ProjectMasksOnBoxes(rcnn::structures::SegmentationMask segmentation_masks, rcnn::structures::BoxList proposals, int discretization_size){
  std::vector<torch::Tensor> masks;
  int M = discretization_size;
  auto device = proposals.get_bbox().device();
  proposals = proposals.Convert("xyxy");
  assert(segmentation_masks.Length() == proposals.Length());

  torch::Tensor proposals_tensor = proposals.get_bbox().to(torch::Device("CPU"));
  for(int i = 0; i < proposals.Length(); ++i){
    rcnn::structures::SegmentationMask cropped_mask = segmentation_masks.Crop(proposals_tensor.select(0, i));
    rcnn::structures::SegmentationMask scaled_mask = cropped_mask.Resize({M, M});
    torch::Tensor mask = scaled_mask.GetMaskTensor();
    masks.push_back(mask);
  }

  if(masks.size() == 0)
    return torch::empty({0}).to(torch::kF32).to(device);
  return torch::stack(masks, 0).to(device).to(torch::kF32);
    
}

MaskRCNNLossComputation::MaskRCNNLossComputation(Matcher* proposal_matcher, int discretization_size) 
                                                :proposal_matcher_(proposal_matcher), 
                                                 discretization_size_(discretization_size){};

MaskRCNNLossComputation::~MaskRCNNLossComputation(){
  if(proposal_matcher_)
    delete proposal_matcher_;
}

MaskRCNNLossComputation::MaskRCNNLossComputation(const MaskRCNNLossComputation& other){
  if(proposal_matcher_)
    delete proposal_matcher_;

  proposal_matcher_ = new Matcher(*other.proposal_matcher_);
  discretization_size_ = other.discretization_size_;
}

MaskRCNNLossComputation::MaskRCNNLossComputation(MaskRCNNLossComputation&& other){
  if(proposal_matcher_)
    delete proposal_matcher_;

  proposal_matcher_ = other.proposal_matcher_;
  discretization_size_ = other.discretization_size_;
  other.proposal_matcher_ = nullptr;
}

MaskRCNNLossComputation MaskRCNNLossComputation::operator=(const MaskRCNNLossComputation& other){
  if(proposal_matcher_)
    delete proposal_matcher_;

  proposal_matcher_ = new Matcher(*other.proposal_matcher_);
  discretization_size_ = other.discretization_size_;

  return *this;
}

MaskRCNNLossComputation MaskRCNNLossComputation::operator=(MaskRCNNLossComputation&& other){
  if(proposal_matcher_)
    delete proposal_matcher_;

  proposal_matcher_ = other.proposal_matcher_;
  discretization_size_ = other.discretization_size_;
  other.proposal_matcher_ = nullptr;

  return *this;
}

rcnn::structures::BoxList MaskRCNNLossComputation::MatchTargetsToProposals(rcnn::structures::BoxList proposal, rcnn::structures::BoxList target){
  torch::Tensor match_quality_matrix = rcnn::structures::BoxList::BoxListIOU(target, proposal);
  torch::Tensor matched_idxs = (*proposal_matcher_)(match_quality_matrix);

  target = target.CopyWithFields(std::vector<std::string>{"labels", "masks"});
  rcnn::structures::BoxList matched_targets = target[matched_idxs.clamp(0)];
  matched_targets.AddField("matched_idxs", matched_idxs);
  return matched_targets;
}


std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> MaskRCNNLossComputation::PrepareTargets(std::vector<rcnn::structures::BoxList> proposals, 
                                                                                std::vector<rcnn::structures::BoxList> targets){
  std::vector<torch::Tensor> labels, masks;

  assert(proposals.size() == targets.size());
  for(int i = 0; i < proposals.size(); ++i){
    rcnn::structures::BoxList matched_targets = MatchTargetsToProposals(proposals[i], targets[i]);
    torch::Tensor matched_idxs = matched_targets.GetField("matched_idxs");
    torch::Tensor labels_per_image = matched_targets.GetField("labels");
    labels_per_image = labels_per_image.to(torch::kInt64);
    torch::Tensor bg_inds = (matched_idxs == Matcher::BELOW_LOW_THRESHOLD);
    labels_per_image.masked_fill(bg_inds, 0);
    torch::Tensor ignore_inds = (matched_idxs == Matcher::BETWEEN_THRESHOLDS);
    labels_per_image.masked_fill(ignore_inds, -1);

    torch::Tensor positive_inds = torch::nonzero(labels_per_image > 0).squeeze(1);
    rcnn::structures::SegmentationMask* segmentation_mask = matched_targets.GetMasksField("masks");
    rcnn::structures::SegmentationMask segmentation_mask_positive = (*segmentation_mask)[positive_inds];

    rcnn::structures::BoxList positive_proposals = proposals[i][positive_inds];
    torch::Tensor masks_per_image = ProjectMasksOnBoxes(segmentation_mask_positive, positive_proposals, discretization_size_);
    labels.push_back(labels_per_image);
    masks.push_back(masks_per_image);
  }

  return std::make_pair(labels, masks);
}

torch::Tensor MaskRCNNLossComputation::operator()(std::vector<rcnn::structures::BoxList> proposals, 
                                                  torch::Tensor mask_logits, 
                                                  std::vector<rcnn::structures::BoxList> targets)
{
  std::vector<torch::Tensor> labels, masks;
  std::tie(labels, masks) = PrepareTargets(proposals, targets);

  torch::Tensor labels_vec = rcnn::layers::cat(labels), mask_targets_vec = rcnn::layers::cat(masks);
  torch::Tensor positive_inds = torch::nonzero(labels_vec > 0).squeeze(1);
  torch::Tensor labels_pos = labels_vec.index_select(0, positive_inds);
  if(mask_targets_vec.numel() == 0)
    return mask_logits.sum() * 0;

  return torch::binary_cross_entropy_with_logits(mask_logits.index_select(0, positive_inds).index_select(1, labels_pos), 
                                                 mask_targets_vec, {}, {}, Reduction::Mean);

}

MaskRCNNLossComputation MakeROIMaskLossEvaluator(){
  Matcher* matcher = new Matcher(
    rcnn::config::GetCFG<float>({"MODEL", "ROI_HEADS", "FG_IOU_THRESHOLD"}),
    rcnn::config::GetCFG<float>({"MODEL", "ROI_HEADS", "BG_IOU_THRESHOLD"}),
    false
  );

  return MaskRCNNLossComputation(matcher, rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_MASK_HEAD", "RESOLUTION"}));
}

}
}