#pragma once
#include <torch/torch.h>
#include "bounding_box.h"
#include "box_coder.h"


namespace rcnn{
namespace modeling{

  class RPNPostProcessorImpl : public torch::nn::Module{
    public:
      RPNPostProcessorImpl(const int pre_nms_top_n, const int post_nms_top_n, const float nms_thresh, const int min_size, BoxCoder& box_coder, const int fpn_post_nms_top_n);
      std::vector<rcnn::structures::BoxList> AddGtProposals(std::vector<rcnn::structures::BoxList> proposals, std::vector<rcnn::structures::BoxList> targets);
      std::vector<rcnn::structures::BoxList> ForwardForSingleFeatureMap(std::vector<rcnn::structures::BoxList> anchors, torch::Tensor objectness, torch::Tensor box_regression);
      std::vector<rcnn::structures::BoxList> SelectOverAllLayers(std::vector<rcnn::structures::BoxList> boxlists);
      std::vector<rcnn::structures::BoxList> forward(std::vector<std::vector<rcnn::structures::BoxList>> anchors, std::vector<torch::Tensor> objectness, std::vector<torch::Tensor> box_regression, std::vector<rcnn::structures::BoxList> targets);
      std::vector<rcnn::structures::BoxList> forward(std::vector<std::vector<rcnn::structures::BoxList>> anchors, std::vector<torch::Tensor> objectness, std::vector<torch::Tensor> box_regression);

    private:
      int64_t pre_nms_top_n_;
      int64_t post_nms_top_n_;
      float nms_thresh_;
      int64_t min_size_;
      BoxCoder box_coder_;
      int64_t fpn_post_nms_top_n_;
  };
  
  TORCH_MODULE(RPNPostProcessor);

  RPNPostProcessor MakeRPNPostprocessor(BoxCoder& rpn_box_coder, bool is_train);
}
}