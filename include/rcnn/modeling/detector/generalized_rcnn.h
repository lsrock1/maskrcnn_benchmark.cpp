#pragma once
#include <torch/torch.h>
#include <cassert>

#include <image_list.h>
#include <bounding_box.h>
#include <defaults.h>


#include "backbone/backbone.h"
#include "rpn/rpn.h"
#include "roi_heads/roi_heads.h"


namespace rcnn{
namespace modeling{

class GeneralizedRCNNImpl : public torch::nn::Module{

public:
  GeneralizedRCNNImpl();
  
  template<typename T>
  T forward(std::vector<torch::Tensor> images, std::vector<rcnn::structures::BoxList> targets);

  template<typename T>
  T forward(rcnn::structures::ImageList images, std::vector<rcnn::structures::BoxList> targets);
  
  std::vector<rcnn::structures::BoxList> forward(std::vector<torch::Tensor> images);
  std::vector<rcnn::structures::BoxList> forward(rcnn::structures::ImageList images);

private:
  Backbone backbone;
  RPNModule rpn;
  CombinedROIHeads roi_heads;
};

TORCH_MODULE(GeneralizedRCNN);

}
}