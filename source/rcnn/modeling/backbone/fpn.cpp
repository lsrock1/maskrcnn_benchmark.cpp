#include "fpn.h"


namespace rcnn{
namespace modeling{

  FPNImpl::FPNImpl(bool use_relu, const std::vector<int64_t> in_channels_list, int64_t out_channels){
    for(int i = 0; i < in_channels_list.size(); ++i){
      inner_blocks_.emplace_back(register_module("fpn_inner"+std::to_string(i+1), rcnn::layers::ConvWithKaimingUniform(use_relu, in_channels_list[i], out_channels, 1)));
      layer_blocks_.emplace_back(register_module("fpn_layers"+std::to_string(i+1), rcnn::layers::ConvWithKaimingUniform(use_relu, out_channels, out_channels, 3, 1)));
    }
  };

  std::deque<torch::Tensor> FPNImpl::forward(std::vector<torch::Tensor>& x){
    std::deque<torch::Tensor> results;
    torch::Tensor inner_top_down;
    torch::Tensor inner_lateral;
    torch::Tensor last_inner = inner_blocks_[3]->forward(x[3]);
    results.push_front(layer_blocks_[3]->forward(last_inner));
    for(int i = inner_blocks_.size()-2; i >= 0; --i){
      inner_top_down = torch::upsample_bilinear2d(last_inner, {last_inner.size(2)*2, last_inner.size(3)*2}, false);
      inner_lateral = inner_blocks_[i]->forward(x[i]);
      last_inner = inner_top_down + inner_lateral;
      results.push_front(layer_blocks_[i]->forward(last_inner));
    }
    
    return results;
  }

  FPNLastMaxPoolImpl::FPNLastMaxPoolImpl(bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels)
                                        :fpn_(register_module("fpn", FPN(use_relu, in_channels_list, out_channels))),
                                         last_level_(register_module("max_pooling", LastLevelMaxPool())){};

  std::deque<torch::Tensor> FPNLastMaxPoolImpl::forward(std::vector<torch::Tensor>& x){
    std::deque<torch::Tensor> results = fpn_->forward(x);
    results.push_back(last_level_->forward(results.back()));
    return results;
  }

  torch::Tensor LastLevelMaxPoolImpl::forward(torch::Tensor& x){
    return torch::max_pool2d(x, 1, 2, 0);
  }
}
}