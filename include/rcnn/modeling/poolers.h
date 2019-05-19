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

class PoolerImpl : public torch::nn::Module{
  public:
    PoolerImpl(std::pair<int, int> output_size, std::vector<float> scales, int sampling_ratio);
    torch::Tensor ConvertToROIFormat(std::vector<rcnn::structures::BoxList> boxes);
    torch::Tensor forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> boxes);

  public:
    std::vector<rcnn::layers::ROIAlign> poolers_;
    std::pair<int, int> output_size_;
    LevelMapper map_levels_;
};

TORCH_MODULE(Pooler);

Pooler MakePooler(std::string head_name);
  
}
}