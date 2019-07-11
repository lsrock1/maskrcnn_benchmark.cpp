#pragma once
#include <torch/torch.h>
#include <set>

#include "roi_heads/box_head/box_head.h"
#include "roi_heads/mask_head/mask_head.h"


namespace rcnn{
namespace modeling{

class CombinedROIHeadsImpl : public torch::nn::Module{

public:
  CombinedROIHeadsImpl(std::set<std::string> heads, int64_t in_channels);
  std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
  std::tuple<torch::Tensor, std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(std::vector<torch::Tensor> features, std::vector<rcnn::structures::BoxList> proposals);
  std::shared_ptr<CombinedROIHeadsImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

private:
  ROIMaskHead mask{nullptr};
  ROIBoxHead box{nullptr};
  int64_t in_channels_;
};

TORCH_MODULE(CombinedROIHeads);

CombinedROIHeads BuildROIHeads(int64_t out_channels);

}
}