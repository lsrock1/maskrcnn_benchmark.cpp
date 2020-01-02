#include "roi_heads/box_head/roi_box_feature_extractors.h"
#include "make_layers.h"
#include "defaults.h"
#include <cmath>


namespace rcnn {
namespace modeling {

ResNet50Conv5ROIFeatureExtractorImpl::ResNet50Conv5ROIFeatureExtractorImpl(int64_t in_channels)
  :pooler_(register_module("pooler", MakePooler("ROI_BOX_HEAD"))),
   head_(register_module("head",
    ResNetHead(
      std::vector<ResNetImpl::StageSpec>{ResNetImpl::StageSpec(4, 3, false)},
      rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "NUM_GROUPS"}),
      rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "WIDTH_PER_GROUP"}),
      rcnn::config::GetCFG<bool>({"MODEL", "RESNETS", "STRIDE_IN_1X1"}),
      0,
      rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "RES2_OUT_CHANNELS"}),
      rcnn::config::GetCFG<int64_t>({"MODEL", "RESNETS", "RES5_DILATION"})
    )
  )),
  out_channels_(head_->out_channels()) {}

torch::Tensor ResNet50Conv5ROIFeatureExtractorImpl::forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> proposals){
  torch::Tensor output = pooler_->forward(x, proposals);
  output = head_->forward(output);
  return output;
}

int64_t ResNet50Conv5ROIFeatureExtractorImpl::out_channels() const{
  return out_channels_;
}

FPN2MLPFeatureExtractorImpl::FPN2MLPFeatureExtractorImpl(int64_t in_channels)
  :pooler_(register_module("pooler", MakePooler("ROI_BOX_HEAD"))),
  fc6_(register_module("fc6", layers::MakeFC(
      in_channels * pow(rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "POOLER_RESOLUTION"}), 2),
      rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "MLP_HEAD_DIM"})
    ))),//fc6
  fc7_(register_module("fc7", layers::MakeFC(
      rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "MLP_HEAD_DIM"}),
      rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "MLP_HEAD_DIM"})
    ))),//fc7
  out_channels_(rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "MLP_HEAD_DIM"})){}

torch::Tensor FPN2MLPFeatureExtractorImpl::forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> proposals){
  torch::Tensor output = pooler_->forward(x, proposals);
  output = output.reshape({output.size(0), -1});
  output = fc6_->forward(output).relu_();
  output = fc7_->forward(output).relu_();
  return output;
}

int64_t FPN2MLPFeatureExtractorImpl::out_channels() const {
  return out_channels_;
}

torch::nn::Sequential FPNXconv1fcFeatureExtractorImpl::make_xconvs(int64_t in_channels) {
  torch::nn::Sequential xconvs;
  int64_t conv_head_dim = rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "CONV_HEAD_DIM"});
  int64_t num_stacked_convs = rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "NUM_STACKED_CONVS"});
  int64_t dilation = rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "DILATION"});
  for (int64_t i = 0; i < num_stacked_convs; ++i) {
    xconvs->push_back(
      torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_channels, conv_head_dim, 3).stride(1).padding(dilation).dilation(dilation).with_bias(true)
      )
    );
    in_channels = conv_head_dim;
    xconvs->push_back(torch::nn::Functional(torch::relu));
  }
  for (auto& param : xconvs->named_parameters()) {
    if (param.key().find("weight") != std::string::npos) {
      torch::nn::init::normal_(param.value(), 0.01);
    }
    else if (param.key().find("bias") != std::string::npos) {
      torch::nn::init::constant_(param.value(), 0);
    }
  }
  return xconvs;
}

FPNXconv1fcFeatureExtractorImpl::FPNXconv1fcFeatureExtractorImpl(int64_t in_channels)
  :pooler_(register_module("pooler", MakePooler("ROI_BOX_HEAD"))),
   xconvs_(register_module("xconvx", make_xconvs(in_channels))),
   fc6_(register_module("fc6", layers::MakeFC(
      rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "CONV_HEAD_DIM"}) * pow(rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "POOLER_RESOLUTION"}), 2),
      rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "MLP_HEAD_DIM"})
    ))),
  out_channels_(rcnn::config::GetCFG<int64_t>({"MODEL", "ROI_BOX_HEAD", "MLP_HEAD_DIM"})) {}

torch::Tensor FPNXconv1fcFeatureExtractorImpl::forward(std::vector<torch::Tensor> x, std::vector<rcnn::structures::BoxList> proposals){
  torch::Tensor output = pooler_->forward(x, proposals);
  output = xconvs_->forward(output);
  output = output.reshape({output.size(0), -1});
  output = fc6_->forward(output).relu_();
  return output;
}

int64_t FPNXconv1fcFeatureExtractorImpl::out_channels() const {
  return out_channels_;
}

std::pair<torch::nn::Sequential, int64_t> MakeROIBoxFeatureExtractor(int64_t in_channels) {
  torch::nn::Sequential extractor;
  int64_t out_channels;
  std::string name = rcnn::config::GetCFG<std::string>({"MODEL", "ROI_BOX_HEAD", "FEATURE_EXTRACTOR"});
  if (name.compare("ResNet50Conv5ROIFeatureExtractor") == 0) {
    auto model = ResNet50Conv5ROIFeatureExtractor(in_channels);
    extractor->push_back(model);
    out_channels = model->out_channels();
  }
  else if (name.compare("FPN2MLPFeatureExtractor") == 0) {
    auto model = FPN2MLPFeatureExtractor(in_channels);
    extractor->push_back(model);
    out_channels = model->out_channels();
  }
  else if (name.compare("FPNXconv1fcFeatureExtractor") == 0) {
    auto model = FPNXconv1fcFeatureExtractor(in_channels);
    extractor->push_back(model);
    out_channels = model->out_channels();
  }
  else {
    assert(false);
  }
  return std::make_pair(extractor, out_channels);
}

} // namespace modeling
} // namespace rcnn
