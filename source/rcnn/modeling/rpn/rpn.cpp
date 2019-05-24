#include "rpn/rpn.h"
#include "defaults.h"


namespace rcnn{
namespace modeling{

RPNHeadImpl::RPNHeadImpl(int64_t in_channels, int64_t num_anchors)
  :conv_(register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1)))),
  cls_logits_(register_module("cls_logits", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_anchors, 1)))),
  bbox_pred_(register_module("bbox_pred", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_anchors * 4, 1))))
{
  for(auto &param : this->named_parameters()){
    if(param.key().find("weight") != std::string::npos) {
      torch::nn::init::normal_(param.value(), 0, 0.01);
    }
    else{
      torch::nn::init::zeros_(param.value());
    }
  }
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> RPNHeadImpl::forward(std::vector<torch::Tensor> x){
  std::vector<torch::Tensor> logits;
  std::vector<torch::Tensor> bbox_reg;
  for(auto& feature: x){
    torch::Tensor t = conv_->forward(feature).relu_();
    logits.push_back(cls_logits_->forward(t));
    bbox_reg.push_back(bbox_pred_->forward(t));
  }
  std::make_pair(logits, bbox_reg);
}

RPNModuleImpl::RPNModuleImpl(int64_t in_channels)
  :anchor_generator_(MakeAnchorGenerator()),
  head_(RPNHead(in_channels, anchor_generator_->NumAnchorsPerLocation()[0])),
  rpn_box_coder_(BoxCoder(std::vector<float>{1.0, 1.0, 1.0, 1.0})),
  box_selector_train_(MakeRPNPostprocessor(rpn_box_coder_, /*is_train=*/true)),
  box_selector_test_(MakeRPNPostprocessor(rpn_box_coder_, /*is_train=*/false)),
  loss_evaluator_(MakeRPNLossEvaluator(rpn_box_coder_)),
  rpn_only_(rcnn::config::GetCFG<bool>({"MODEL", "RPN_ONLY"})){}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> RPNModuleImpl::forward(rcnn::structures::ImageList& images, std::vector<torch::Tensor>& features, std::vector<rcnn::structures::BoxList>& targets){
  //given targets
  std::vector<torch::Tensor> objectness, rpn_box_regression;
  std::tie(objectness, rpn_box_regression) = head_(features);
  std::vector<std::vector<rcnn::structures::BoxList>> anchors = anchor_generator_(images, features);

  if(is_training()){
    forward_train(anchors, objectness, rpn_box_regression, targets);
  }
  else{
    forward_test(anchors, objectness, rpn_box_regression);
  }
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> RPNModuleImpl::forward(rcnn::structures::ImageList& images, std::vector<torch::Tensor>& features){
  //no targets
  std::vector<torch::Tensor> objectness, rpn_box_regression;
  std::tie(objectness, rpn_box_regression) = head_(features);
  std::vector<std::vector<rcnn::structures::BoxList>> anchors = anchor_generator_(images, features);
  
  return forward_test(anchors, objectness, rpn_box_regression);
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> RPNModuleImpl::forward_train(std::vector<std::vector<rcnn::structures::BoxList>>& anchors, std::vector<torch::Tensor>& objectness, std::vector<torch::Tensor>& rpn_box_regression, std::vector<rcnn::structures::BoxList>& targets){
  std::vector<rcnn::structures::BoxList> boxes;
  torch::Tensor loss_objectness, loss_rpn_box_reg;
  std::map<std::string, torch::Tensor> losses;
  if(rpn_only_){
    //cat anchors per image [not in original implementation]
    for(int i = 0; i < anchors.size(); ++i)
      boxes.push_back(rcnn::structures::BoxList::CatBoxList(anchors[i]));
  }
  else{
    //no_grad bracket
    {
      torch::NoGradGuard guard;
      boxes = box_selector_train_->forward(
        anchors, objectness, rpn_box_regression, targets
      );
    }
  }
  std::tie(loss_objectness, loss_rpn_box_reg) = loss_evaluator_(anchors, objectness, rpn_box_regression, targets);
  losses["loss_objectness"] = loss_objectness;
  losses["loss_rpn_box_reg"] = loss_rpn_box_reg;
  return std::make_pair(boxes, losses);
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> RPNModuleImpl::forward_test(std::vector<std::vector<rcnn::structures::BoxList>>& anchors, std::vector<torch::Tensor>& objectness, std::vector<torch::Tensor>& rpn_box_regression){
  std::vector<rcnn::structures::BoxList> boxes = box_selector_train_->forward(anchors, objectness, rpn_box_regression);
  std::map<std::string, torch::Tensor> losses;
  if(rpn_only_){
    for(auto& box: boxes){
      //get index and sort box
      box = box[std::get<1>(box.GetField("objectness").sort(/*dim=*/-1, true))];
    }
  }
  return std::make_pair(boxes, losses);
}

RPNModule BuildRPN(int64_t in_channels){
  return RPNModule(in_channels);
}

}
}