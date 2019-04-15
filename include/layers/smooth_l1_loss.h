#pragma once
#include <torch/torch.h>

torch::Tensor smooth_l1_loss(torch::Tensor input, torch::Tensor target, float beta=1. / 9, bool size_average=true){
    auto n = torch::abs(input - target);
    cond =  n < beta;
    loss = torch::where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if(size_average){
        return loss.mean();
    }
    return loss.sum(); 
}