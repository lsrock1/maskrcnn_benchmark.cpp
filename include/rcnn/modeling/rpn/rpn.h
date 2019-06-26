#pragma once
#include <torch/torch.h>
#include <map>
#include <vector>
#include "bounding_box.h"
#include "rpn/inference.h"
#include "box_coder.h"
#include "matcher.h"
#include "rpn/anchor_generator.h"
#include "rpn/loss.h"


namespace rcnn{
namespace modeling{
  
class RPNHeadImpl : public torch::nn::Module{
  public:
    RPNHeadImpl(int64_t in_channels, int64_t num_anchors);
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward(std::vector<torch::Tensor> x); 

  private:
    torch::nn::Conv2d conv_;
    torch::nn::Conv2d cls_logits_;
    torch::nn::Conv2d bbox_pred_;
};

TORCH_MODULE(RPNHead);

class RPNModuleImpl : public torch::nn::Module{
  public:
    RPNModuleImpl(int64_t in_channels);
    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(rcnn::structures::ImageList& images, std::vector<torch::Tensor>& features, std::vector<rcnn::structures::BoxList> targets);
    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(rcnn::structures::ImageList& images, std::vector<torch::Tensor>& features);


  private:
    AnchorGenerator anchor_generator_;
    RPNHead head_;
    BoxCoder rpn_box_coder_;
    RPNPostProcessor box_selector_train_;
    RPNPostProcessor box_selector_test_;
    RPNLossComputation loss_evaluator_;
    bool rpn_only_;

    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward_train(std::vector<std::vector<rcnn::structures::BoxList>>& anchors, std::vector<torch::Tensor>& objectness, std::vector<torch::Tensor>& rpn_box_regression, std::vector<rcnn::structures::BoxList> targets);
    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward_test(std::vector<std::vector<rcnn::structures::BoxList>>& anchors, std::vector<torch::Tensor>& objectness, std::vector<torch::Tensor>& rpn_box_regression);
};

TORCH_MODULE(RPNModule);

RPNModule BuildRPN(int64_t in_channels);

}
}