#include "vision_cpu.h"


namespace rcnn{
namespace layers{

torch::Tensor match_proposals_cpu(torch::Tensor match_quality_matrix, bool allow_low_quality_matches, 
                                float low_th, float high_th){
  torch::Tensor matched_vals, matches, all_matches;
  std::tie(matched_vals, matches) = match_quality_matrix.max(/*dim=*/0);
  if(allow_low_quality_matches)
    all_matches = matches.clone();
  torch::Tensor upper_threshold = matched_vals >= high_th;

  torch::Tensor below_low_threshold = matched_vals < low_th;
  torch::Tensor between_thresholds = (matched_vals >= low_th).__and__(matched_vals < high_th);

  matches.masked_fill_(below_low_threshold, -1);
  matches.masked_fill_(between_thresholds, -2);

  if(allow_low_quality_matches){
    torch::Tensor highest_quality_foreach_gt, gt_pred_pairs_of_highest_quality, pred_inds_to_update;

    highest_quality_foreach_gt = std::get<0>(match_quality_matrix.max(/*dim=*/1));
    gt_pred_pairs_of_highest_quality = torch::nonzero(match_quality_matrix == highest_quality_foreach_gt.unsqueeze(1));
    pred_inds_to_update = gt_pred_pairs_of_highest_quality.select(/*dim=*/1, /*index=*/1);//.unsqueeze(1);
    matches.index_copy_(0, pred_inds_to_update, all_matches.index_select(0, pred_inds_to_update));
  }
    
  return matches;
}

}
}