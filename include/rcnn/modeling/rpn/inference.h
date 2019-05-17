#pragma once
#include <torch/torch.h>
#include "bounding_box.h"
#include "box_coder.h"


namespace rcnn{
namespace modeling{

  class RPNPostProcessorImpl : public torch::nn::Module{
    public:
      RPNPostProcessorImpl(int pre_nms_top_n, int post_nms_top_n, float nms_thresh, int min_size, BoxCoder box_coder, int fpn_post_nms_top_n);
      std::vector<rcnn::structures::BoxList> AddGtProposals(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
      std::vector<rcnn::structures::BoxList> ForwardForSingleFeatureMap(std::vector<rcnn::structures::BoxList> anchors, torch::Tensor objectness, torch::Tensor box_regression);

    private:
      int pre_nms_top_n_;
      int post_nms_top_n_;
      float nms_thresh_;
      int min_size_;
      BoxCoder box_coder_;
      int fpn_post_nms_top_n_;
  }
  
  TORCH_MODULE(RPNPostProcessor)
}
}