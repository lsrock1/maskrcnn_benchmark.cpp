#pragma once
#include <torch/torch.h>
#include <string>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>


namespace rcnn{
namespace layers{
//Reference from https://gist.github.com/mikehamer/df0af5ec7ff98d3cae487975d0c921df
torch::Tensor _NewEmptyTensorOp(const torch::Tensor x, torch::IntArrayRef new_shape);

struct _NewEmptyTensorOpBackward : public torch::autograd::Node{
  torch::IntArrayRef shape;
  torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;
};

class Conv2dImpl : public torch::nn::Conv2dImpl{
  public:
    Conv2dImpl(torch::nn::Conv2dOptions conv2dOptions): torch::nn::Conv2dImpl(conv2dOptions){};
    torch::Tensor forward(const Tensor& input);
};

TORCH_MODULE(Conv2d);

void check_size_scale_factor(int dim);
torch::IntArrayRef output_size(int dim);
torch::Tensor interpolate(torch::Tensor input, torch::IntArrayRef size/* , float scale_factor, std::string mode, bool align_corners*/);

}//layers
}//rcnn