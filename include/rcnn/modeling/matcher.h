#pragma once
#include <torch/torch.h>


namespace rcnn{
namespace modeling{

class Matcher{
  public:
    static const int BELOW_LOW_THRESHOLD = -1;
    static const int BETWEEN_THRESHOLDS = -2;
    torch::Tensor operator()(torch::Tensor& match_quality_matrix);
    Matcher(float high_threshold, float low_threshold, bool allow_low_quality_matches = false);
    void SetLowQualityMatches(torch::Tensor& matches, torch::Tensor& all_matches, torch::Tensor& match_quality_matrix);

  private:
    float high_threshold_;
    float low_threshold_;
    bool allow_low_quality_matches_;
};

}
}