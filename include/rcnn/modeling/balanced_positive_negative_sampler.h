#pragma once
#include <torch/torch.h>
#include <vector>


namespace rcnn{
namespace modeling{

class BalancedPositiveNegativeSampler{
  public:
    BalancedPositiveNegativeSampler(int64_t batch_size_per_image, double positive_fraction);
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> operator()(std::vector<torch::Tensor> matched_idxs);

  private:
    // int batch_size_per_image_;
    // double positive_fraction_;
    int64_t num_pos_;
    int64_t num_neg_;
};
}
}//rcnn