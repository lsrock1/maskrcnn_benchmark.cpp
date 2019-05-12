#pragma once
#include <torch/torch.h>


namespace rcnn{
namespace layers{
//modeling/make_layers.py
torch::nn::Sequential ConvWithKaimingUniform(/*NO GN use_gn=false, */bool use_relu, int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride=1, int64_t dilation=1);
torch::nn::Linear MakeFC(int64_t dim_in, int64_t hidden_dim);
torch::nn::Sequential MakeConv3x3(int64_t in_channels, int64_t out_channels, int64_t dilation=1, int64_t stride=1, /*use_gn, */bool use_relu=false, bool kaiming_init=true);
}
}