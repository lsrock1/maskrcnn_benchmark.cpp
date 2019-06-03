#pragma once
#include <torch/torch.h>


namespace rcnn{
namespace modeling{

class FastRCNNPredictorImpl : public torch::nn::Module{
  public:
    FastRCNNPredictorImpl(int64_t in_channels);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

  private:
    torch::nn::Linear cls_score_;
    torch::nn::Linear bbox_pred_;
};

TORCH_MODULE(FastRCNNPredictor);

class FPNPredictorImpl : public torch::nn::Module{
  public:
    FPNPredictorImpl(int64_t in_channels);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

  private:
    torch::nn::Linear cls_score_;
    torch::nn::Linear bbox_pred_;
};

TORCH_MODULE(FPNPredictor);

torch::nn::Sequential MakeROIBoxPredictor(int64_t in_channels);

}
}