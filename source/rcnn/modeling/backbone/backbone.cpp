#include "backbone/backbone.h"

namespace rcnn{
namespace modeling{

  template<typename Backbone, typename FPNType>
  BackboneImpl<Backbone, FPNType>::BackboneImpl(Backbone body){
    body_ = register_module("body", body);
    //base net must implement
    //get bottom channels
    //get out channels
    int64_t bottom_channels = body_->get_bottom_channels();
    if(body_ -> get_is_fpn()){
      //this part is hard coded from resnet
      std::vector<int64_t> vec{bottom_channels, bottom_channels*2, bottom_channels*4, bottom_channels*8};
      fpn_ = register_module("fpn", FPNType(/*use_relu*/rcnn::config::GetCFG<bool>({"MODEL", "FPN", "USE_RELU"}), vec, body_->get_out_channels()));
    }
  }

  template<typename Backbone, typename FPNType>
  std::vector<torch::Tensor> BackboneImpl<Backbone, FPNType>::forward(torch::Tensor x){
    
    if(fpn_){
      std::vector<torch::Tensor> body_results = body_->forward_fpn(x);
      return fpn_ -> forward(body_results);
    }
    else{
      std::vector<torch::Tensor> results;
      results.push_back(body_->forward(x));
      return results;
    }
  }
  
  torch::nn::Sequential BuildBackBone(){
    torch::nn::Sequential result;
    rcnn::config::CFGString backbone_name_wrapper = rcnn::config::GetCFG<rcnn::config::CFGString>({"MODEL", "BACKBONE", "CONV_BODY"});
    std::string backbone_name = backbone_name_wrapper.get();
    if(rcnn::modeling::ResBackbonesMap().count(backbone_name)){
      auto body = ResNet(rcnn::modeling::ResBackbonesMap().find(backbone_name)->second);
      result->push_back(ResBackbone(body));
    }
    else{
      throw "Backbone not found";
    }
    return result;
  }
}
}