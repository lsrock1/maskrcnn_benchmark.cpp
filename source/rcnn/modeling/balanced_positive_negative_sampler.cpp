#include "balanced_positive_negative_sampler.h"
#include <iostream>


namespace rcnn{
namespace modeling{

BalancedPositiveNegativeSampler::BalancedPositiveNegativeSampler(int64_t batch_size_per_image, float positive_fraction)
    :num_pos_((int64_t) (batch_size_per_image * positive_fraction)),
     num_neg_(batch_size_per_image - num_pos_){}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> BalancedPositiveNegativeSampler::operator()(std::vector<torch::Tensor> matched_idxs){
  std::vector<torch::Tensor> pos_idx;
  std::vector<torch::Tensor> neg_idx;
  int64_t num_pos;
  int64_t num_neg;
  torch::Tensor pos_perm;
  torch::Tensor neg_perm;
  torch::Tensor pos_idx_per_image;
  torch::Tensor neg_idx_per_image;
  torch::Tensor pos_idx_per_image_mask;
  torch::Tensor neg_idx_per_image_mask;
  for(torch::Tensor& matched_idxs_per_image : matched_idxs){
    //list of positive or negative idxs
    torch::Tensor positive = torch::nonzero(matched_idxs_per_image >= 1).squeeze(1);
    torch::Tensor negative = torch::nonzero(matched_idxs_per_image == 0).squeeze(1);

    //prevent from over indexing
    num_pos = std::min(num_pos_, positive.numel());
    num_neg = std::min(num_neg_, negative.numel());

    pos_perm = torch::randperm(positive.numel(), torch::TensorOptions().device(positive.device()).dtype(positive.dtype())).slice(/*dim=*/0, /*start=*/0, /*end=*/num_pos);
    neg_perm = torch::randperm(negative.numel(), torch::TensorOptions().device(negative.device()).dtype(negative.dtype())).slice(/*dim=*/0, /*start=*/0, /*end=*/num_neg);
    pos_idx_per_image = positive.index_select(/*dim=*/0, pos_perm);
    neg_idx_per_image = negative.index_select(0, neg_perm);

    pos_idx_per_image_mask = torch::zeros_like(matched_idxs_per_image, torch::TensorOptions().dtype(torch::kInt8).device(matched_idxs_per_image.device()));
    neg_idx_per_image_mask = torch::zeros_like(matched_idxs_per_image, torch::TensorOptions().dtype(torch::kInt8).device(matched_idxs_per_image.device()));

    pos_idx_per_image_mask.index_fill_(0, pos_idx_per_image, 1);
    neg_idx_per_image_mask.index_fill_(0, neg_idx_per_image, 1);
    pos_idx.push_back(std::move(pos_idx_per_image_mask));
    neg_idx.push_back(std::move(neg_idx_per_image_mask));
  }
  return std::make_pair(pos_idx, neg_idx);
}

}
}//rcnn