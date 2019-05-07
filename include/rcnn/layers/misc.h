#pragma once
#include <torch/torch.h>
#include <torch/nn/modules/conv.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

namespace rcnn{
namespace layers{
//Reference from https://gist.github.com/mikehamer/df0af5ec7ff98d3cae487975d0c921df
torch::Tensor _NewEmptyTensorOp(const torch::Tensor x, torch::IntArrayRef new_shape);

struct _NewEmptyTensorOpBackward : public torch::autograd::Function{
    torch::IntArrayRef shape;

    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override;
};

class Conv2dImpl : public torch::nn::Conv2dImpl{
    public:
        Conv2dImpl(torch::nn::Conv2dOptions conv2dOptions): torch::nn::Conv2dImpl(conv2dOptions){};
        torch::Tensor forward(const Tensor& input);
};

TORCH_MODULE(Conv2d);

//modeling/make_layers.py
torch::nn::Sequential ConvWithKaimingUniform(/*NO GN use_gn=false, */bool use_relu=false);
torch::nn::Linear MakeFC(int64_t dim_in, int64_t hidden_dim);
torch::nn::Sequential MakeConv3x3(int64_t in_channels, int64_t out_channels, int64_t dilation=1, int64_t stride=1, /*use_gn, */bool use_relu=false, bool kaiming_init=true);
}//layers
}//rcnn