// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/torch.h>

namespace rcnn {
namespace layers {
    
torch::Tensor ROIAlign_forward_cpu(const torch::Tensor& input,
                                   const torch::Tensor& rois,
                                   const float spatial_scale,
                                   const int pooled_height,
                                   const int pooled_width,
                                   const int sampling_ratio,
                                   bool aligned);

torch::Tensor nms_cpu(const torch::Tensor& dets, const torch::Tensor& scores, const float threshold);

inline torch::Tensor box_iou_cpu(torch::Tensor area_a, torch::Tensor area_b, torch::Tensor bbox_a, torch::Tensor bbox_b) {
  int TO_REMOVE = 1;
  torch::Tensor lt = torch::max(bbox_a.unsqueeze(1).slice(/*dim=*/2, /*start=*/0, /*end=*/2), bbox_b.slice(1, 0, 2));
  torch::Tensor rb = torch::min(bbox_a.unsqueeze(1).slice(/*dim=*/2, /*start=*/2, /*end=*/4), bbox_b.slice(1, 2, 4));
  torch::Tensor wh = (rb - lt + TO_REMOVE).clamp(0);
  torch::Tensor inter = wh.select(2, 0) * wh.select(2, 1);
  return inter / (area_a.unsqueeze(1) + area_b - inter);
}

inline torch::Tensor box_encode_cpu(torch::Tensor reference_boxes, torch::Tensor proposals, float wx, float wy, float ww, float wh) {
  int TO_REMOVE = 1;
  torch::Tensor ex_widths = proposals.select(1, 2) - proposals.select(1, 0) + TO_REMOVE;
  torch::Tensor ex_heights = proposals.select(1, 3) - proposals.select(1, 1) + TO_REMOVE;
  torch::Tensor ex_ctr_x = proposals.select(1, 0) + 0.5 * ex_widths;
  torch::Tensor ex_ctr_y = proposals.select(1, 1) + 0.5 * ex_heights;

  torch::Tensor gt_widths = reference_boxes.select(1, 2) - reference_boxes.select(1, 0) + TO_REMOVE;
  torch::Tensor gt_heights = reference_boxes.select(1, 3) - reference_boxes.select(1, 1) + TO_REMOVE;
  torch::Tensor gt_ctr_x = reference_boxes.select(1, 0) + 0.5 * gt_widths;
  torch::Tensor gt_ctr_y = reference_boxes.select(1, 1) + 0.5 * gt_heights;

  return torch::stack({
    /*targets_dx*/ wx * (gt_ctr_x - ex_ctr_x) / ex_widths,
    /*targets_dy*/ wy * (gt_ctr_y - ex_ctr_y) / ex_heights,
    /*targets_dw*/ ww * torch::log(gt_widths / ex_widths),
    /*targets_dh*/ wh * torch::log(gt_heights / ex_heights)
  }, /*dim=*/1);
}

inline torch::Tensor match_proposals_cpu(torch::Tensor match_quality_matrix, bool allow_low_quality_matches, float low_th, float high_th) {
  torch::Tensor matched_vals, matches, all_matches;
  std::tie(matched_vals, matches) = match_quality_matrix.max(/*dim=*/0);
  if (allow_low_quality_matches)
    all_matches = matches.clone();
  torch::Tensor upper_threshold = matched_vals >= high_th;

  torch::Tensor below_low_threshold = matched_vals < low_th;
  torch::Tensor between_thresholds = (matched_vals >= low_th).__and__(matched_vals < high_th);

  matches.masked_fill_(below_low_threshold, -1);
  matches.masked_fill_(between_thresholds, -2);

  if (allow_low_quality_matches) {
    torch::Tensor highest_quality_foreach_gt, gt_pred_pairs_of_highest_quality, pred_inds_to_update;

    highest_quality_foreach_gt = std::get<0>(match_quality_matrix.max(/*dim=*/1));
    gt_pred_pairs_of_highest_quality = torch::nonzero(match_quality_matrix == highest_quality_foreach_gt.unsqueeze(1));
    pred_inds_to_update = gt_pred_pairs_of_highest_quality.select(/*dim=*/1, /*index=*/1);//.unsqueeze(1);
    matches.index_copy_(0, pred_inds_to_update, all_matches.index_select(0, pred_inds_to_update));
  }

  return matches;
}

} // namespace layers
} // namespace rcnn
