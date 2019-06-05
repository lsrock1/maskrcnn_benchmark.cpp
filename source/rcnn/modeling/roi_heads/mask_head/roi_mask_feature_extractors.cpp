#include "roi_heads/mask_head/roi_mask_feature_extractors.h"
#include "defaults.h"
#include "make_layers.h"


namespace rcnn{
namespace modeling{

MaskRCNNFPNFeatureExtractorImpl::MaskRCNNFPNFeatureExtractorImpl(const int64_t in_channels)
  :pooler_(register_module("pooler", MakePooler("ROI_MASK_HEAD")))
{
  std::vector<int64_t> layers = rcnn::config::GetCFG<std::vector<int64_t>>({"MODEL", "ROI_MASK_HEAD", "CONV_LAYERS"});
  int64_t dilation = rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_MASK_HEAD", "DILATION"});
  int64_t next_feature = in_channels;
  for(size_t i = 0; i < layers.size(); ++i){
    std::string layer_name = "mask_fcn" + std::to_string(i+1);
    blocks_.push_back(register_module(layer_name, rcnn::layers::MakeConv3x3(next_feature, layers[i], dilation, 1, true)));
    next_feature = layers[i];
  }
  out_channels_ = layers.back();
}

torch::Tensor MaskRCNNFPNFeatureExtractorImpl::forward(std::vector<torch::Tensor>& x, std::vector<rcnn::structures::BoxList>& proposals){
  torch::Tensor output = pooler_->forward(x, proposals);
  for(auto& block: blocks_)
    output = block->forward(output);
  return output;
}

int64_t MaskRCNNFPNFeatureExtractorImpl::out_channels() const{
  return out_channels_;
}

MaskRCNNFPNFeatureExtractor MakeROIMaskFeatureExtractor(const int64_t in_channels){
  return MaskRCNNFPNFeatureExtractor(in_channels);
}

}
}