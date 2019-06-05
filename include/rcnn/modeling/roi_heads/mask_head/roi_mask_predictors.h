#pragma once
#include <torch/torch.h>
#include "conv2d.h"


namespace rcnn{
namespace modeling{

class MaskRCNNC4PredictorImpl : public torch::nn::Module{
  public:
    MaskRCNNC4PredictorImpl(int64_t in_channels);
    torch::Tensor forward(torch::Tensor x);

  private:
    rcnn::layers::Conv2d conv5_mask;
    rcnn::layers::Conv2d mask_fcn_logits;
};

TORCH_MODULE(MaskRCNNC4Predictor);

class MaskRCNNConv1x1PredictorImpl : public torch::nn::Module{
  public:
    MaskRCNNConv1x1PredictorImpl(int64_t in_channels);
    torch::Tensor forward(torch::Tensor);

  private:
    rcnn::layers::Conv2d mask_fcn_logits;
};

TORCH_MODULE(MaskRCNNConv1x1Predictor);

torch::nn::Sequential MakeROIMaskPredictor(int64_t in_channels);

}
}