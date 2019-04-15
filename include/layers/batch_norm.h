#pragma once
#include <torch/torch.h>

struct FrozenBatchNorm2dImpl : torch::nn::Module {
    FrozenBatchNorm2dImpl(int64_t dimension)
      : weight(register_buffer("weight", torch::ones(dimension))),
        bias(register_buffer("bias", torch::zeros(dimension))),
        mean(register_buffer("running_mean", torch::zeros(dimension))),
        var(register_buffer("running_var", torch::ones(dimension))){}

    torch::Tensor forward(torch::Tensor x){
        // TODO INTEGRATION
        torch::Tensor scale = weight * var.rsqrt();
        bias = bias - mean * scale;
        scale = scale.reshape({1, -1, 1, 1});
        bias = bias.reshape({1, -1, 1, 1});
        return x * scale + bias;
    }

    torch::Tensor weight, bias, mean, var;
};

TORCH_MODULE(FrozenBatchNorm2d);