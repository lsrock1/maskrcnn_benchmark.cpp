#pragma once
#include <torch/torch.h>
#include "bounding_box.h"
#include "poolers.h"


namespace rcnn{
namespace modeling{

class MaskRCNNFPNFeatureExtractorImpl : public torch::nn::Module{
  public:
    MaskRCNNFPNFeatureExtractorImpl(const int64_t in_channels);
    torch::Tensor forward(std::vector<torch::Tensor>& x, std::vector<rcnn::structures::BoxList>& proposals);
    int64_t out_channels() const;

  private:
    Pooler pooler_;
    std::vector<torch::nn::Sequential> blocks_;
    int64_t out_channels_;
};

TORCH_MODULE(MaskRCNNFPNFeatureExtractor);

MaskRCNNFPNFeatureExtractor MakeROIMaskFeatureExtractor(const int64_t in_channels);

}
}