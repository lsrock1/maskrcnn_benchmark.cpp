#include "backbone.h"

namespace rcnn{
namespace modeling{

  template<typename Backbone>
  BackboneImpl<Backbone>::BackboneImpl(Backbone body, FPN fpn)
                    : body_(body),
                      fpn_(fpn){}

  template<typename Backbone>
  BackboneImpl<Backbone>::BackboneImpl(Backbone body)
                    : body_(body){}

  
  torch::Tensor BackboneImpl::forward(torch::Tensor x){
    if(fpn_){
      x = body_->forward_fpn(x);
      return fpn_(x);
    }
    else{
      x = body_->forward(x);
      return x;
    }
  }
}
}