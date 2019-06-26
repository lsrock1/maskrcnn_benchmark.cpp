#include "smooth_l1_loss.h"


namespace rcnn{
namespace layers{
torch::Tensor smooth_l1_loss(torch::Tensor input, torch::Tensor target, float beta, bool size_average){
    torch::Tensor n = torch::abs(input - target);
    auto cond = n < beta;
    cond = cond.to(torch::kF32);
    torch::Tensor loss = cond * ((0.5 * n).pow(2) / beta) + (1 - cond) * (n - 0.5 * beta);
    if(size_average){
        return loss.mean();
    }
    return loss.sum(); 
}
}//layers
}//rcnn
