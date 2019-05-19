#pragma once
#include <torch/torch.h>
#include "bounding_box.h"
#include "roi_align.h"


namespace rcnn{
namespace modeling{
  
class LevelMapper{
  public:
    LevelMapper(int k_min, int k_max, int canonical_scale = 224, int canonical_level = 4, float eps = 1e-6);
    torch::Tensor operator()(std::vector<rcnn::structures::BoxList> boxlists);

  private:
    int k_min_;
    int k_max_;
    int s0_;
    int lvl0_;
    float eps_;
};

class Pooler : public torch::nn::Module{
  public:
    Pooler(std::pair<int, int> output_size, std::vector<float> scales, int sampling_ratio);
    torch::Tensor ConvertToROIFormat(std::vector<rcnn::structures::BoxList> boxes);
    torch::Tensor forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> boxes);

  public:
    std::vector<rcnn::layers::ROIAlign> poolers_;
    std::pair<int, int> output_size_;
    LevelMapper map_levels_;
};

Pooler MakePooler(std::string head_name);
  int resolution = rcnn::config::GetCFG<int>({"MODEL", head_name, "POOLER_RESOLUTION"});
  std::vector<float> scales = rcnn::config::GetCFG<std::vector<float>>({"MODEL", head_name, "POOLER_SCALES"});
  int sampling_ratio = cnn::config::GetCFG<int>({"MODEL", head_name, "POOLER_SAMPLING_RATIO"});
  return Pooler(std::make_pair(resolution, resolution), scales, sampling_ratio);
}
}