#include "backbone/fpn.h"
#include <algorithm>


namespace rcnn{
namespace modeling{

  FPNImpl::FPNImpl(bool use_relu, const std::vector<int64_t> in_channels_list, int64_t out_channels, ConvFunction conv_block){
    for(int i = 0; i < in_channels_list.size(); ++i){
      inner_blocks_.push_back(register_module("fpn_inner"+std::to_string(i+1), conv_block(use_relu, in_channels_list[i], out_channels, 1, 1, 1)));
      layer_blocks_.push_back(register_module("fpn_layer"+std::to_string(i+1), conv_block(use_relu, out_channels, out_channels, 3, 1, 1)));
    }
  };

  std::vector<torch::Tensor> FPNImpl::forward(std::vector<torch::Tensor>& x){
    std::vector<torch::Tensor> results;
    torch::Tensor inner_top_down;
    torch::Tensor inner_lateral;
    torch::Tensor last_inner = inner_blocks_[3]->forward(x[3]);
    results.push_back(layer_blocks_[3]->forward(last_inner));
    for(int i = inner_blocks_.size()-2; i >= 0; --i){
      inner_top_down = torch::upsample_nearest2d(last_inner, {last_inner.size(2)*2, last_inner.size(3)*2});
      inner_lateral = inner_blocks_[i]->forward(x[i]);
      last_inner = inner_top_down + inner_lateral;
      results.push_back(layer_blocks_[i]->forward(last_inner));
    }
    std::reverse(results.begin(), results.end());
    return results;
  }

  FPNLastMaxPoolImpl::FPNLastMaxPoolImpl(bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels, ConvFunction conv_block)
                                        :fpn_(register_module("fpn", FPN(use_relu, in_channels_list, out_channels, conv_block))){};
  
  std::vector<torch::Tensor> FPNLastMaxPoolImpl::forward(std::vector<torch::Tensor> x){
    std::vector<torch::Tensor> results = fpn_->forward(x);
    results.push_back(torch::max_pool2d(results.back(), 1, 2, 0));
    return results;
  }
}
}