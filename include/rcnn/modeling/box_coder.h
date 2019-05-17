#pragma once
#include <torch/torch.h>
#include <cmath>


namespace rcnn{
namespace modeling{
  
  class BoxCoder{
    public:
      BoxCoder(std::vector<float> weights, double bbox_xform_clip = log(1000. / 16));
      torch::Tensor encode(torch::Tensor reference_boxes, torch::Tensor proposals);
      torch::Tensor decode(torch::Tensor rel_codes, torch::Tensor boxes);
    
    private:
      std::vector<float> weights_;
      double bbox_xform_clip_;
  };
}
}