#include "matcher.h"
#include <cassert>


namespace rcnn{
namespace modeling{

Matcher::Matcher(float high_threshold, float low_threshold, bool allow_low_quality_matches)
                :high_threshold_(high_threshold),
                 low_threshold_(low_threshold),
                 allow_low_quality_matches_(allow_low_quality_matches){}

torch::Tensor Matcher::operator()(torch::Tensor& match_quality_matrix){
  //gt x predicted
  //select highest gt per predicted
  assert(match_quality_matrix.numel() != 0);
  torch::Tensor matched_vals, matches, all_matches;
  std::tie(matched_vals, matches) = match_quality_matrix.max(/*dim=*/0);
  if(allow_low_quality_matches_)
    all_matches = matches.clone();
  torch::Tensor upper_threshold = matched_vals >= high_threshold_;

  torch::Tensor below_low_threshold = matched_vals < low_threshold_;
  torch::Tensor between_thresholds = (matched_vals >= low_threshold_).__and__(matched_vals < high_threshold_);

  matches.masked_fill_(below_low_threshold, Matcher::BELOW_LOW_THRESHOLD);
  matches.masked_fill_(between_thresholds, Matcher::BETWEEN_THRESHOLDS);

  if(allow_low_quality_matches_)
    SetLowQualityMatches(matches, all_matches, match_quality_matrix);
  return matches;
}

void Matcher::SetLowQualityMatches(torch::Tensor& matches, torch::Tensor& all_matches, torch::Tensor& match_quality_matrix){
  //select highest predicted per gt additionally
  torch::Tensor highest_quality_foreach_gt, gt_pred_pairs_of_highest_quality, pred_inds_to_update;

  highest_quality_foreach_gt = std::get<0>(match_quality_matrix.max(/*dim=*/1));
  gt_pred_pairs_of_highest_quality = torch::nonzero(match_quality_matrix == highest_quality_foreach_gt.unsqueeze(1));
  pred_inds_to_update = gt_pred_pairs_of_highest_quality.select(/*dim=*/1, /*index=*/1);//.unsqueeze(1);
  matches.index_copy_(0, pred_inds_to_update, all_matches.index_select(0, pred_inds_to_update));
}

}
}