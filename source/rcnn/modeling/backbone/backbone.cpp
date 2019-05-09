#include "backbone.h"

namespace rcnn{
namespace modeling{

  template<typename Backbone>
  BackboneImpl<Backbone>::BackboneImpl(Backbone& body){
    body_ = register_module("body", body);
    int64_t res2_out_channels = body_ -> get_res2_out_channels();
    if(body_ -> get_is_fpn()){
      std::vector<int64_t> vec{res2_out_channels, res2_out_channels*2, res2_out_channels*4, res2_out_channels*8};
      //TODO  
      fpn_ = register_module("fpn", FPNLastMaxPool(/*use_relu*/false, vec, body_ -> get_out_channels()));
    }
  }

  template<typename Backbone>
  std::deque<torch::Tensor> BackboneImpl<Backbone>::forward(torch::Tensor& x){
    
    if(fpn_){
      std::vector<torch::Tensor> body_results = body_->forward_fpn(x);
      return fpn_ -> forward(body_results);
    }
    else{
      std::deque<torch::Tensor> results;
      results.emplace_front(body_->forward(x));
      return results;
    }
  }
}
}