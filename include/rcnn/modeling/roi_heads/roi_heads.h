#pragma once
#include <torch/torch.h>
#include <set>

#include "roi_heads/box_head/box_head.h"
#include "roi_heads/mask_head/mask_head.h"


namespace rcnn{
namespace modeling{

using mh = ROIMaskHead;
using bh = ROIBoxHead;

class CombinedROIHeadsImpl : public torch::nn::Module{

public:
  CombinedROIHeadsImpl(bool box, bool mask, int64_t in_channels);
  std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
  std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals);


private:
  mh mask{nullptr};
  bh box{nullptr};
};

TORCH_MODULE(CombinedROIHeads);

CombinedROIHeads BuildROIHeads(int64_t out_channels);

}
}