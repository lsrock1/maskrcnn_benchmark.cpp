#include "backbone.h"

namespace rcnn{
namespace modeling{

  template<typename Backbone>
  ModelImpl<Backbone>::ModelImpl(Backbone body, FPN fpn)
                    : body_(body),
                      fpn_(fpn){}

  template<typename Backbone>
  ModelImpl<Backbone>::ModelImpl(Backbone body)
                    : body_(body){}

  
  torch::Tensor ModelImpl::forward(torch::Tensor x){
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