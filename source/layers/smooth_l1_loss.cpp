#include <layers/smooth_l1_loss.h>


torch::Tensor smooth_l1_loss(torch::Tensor input, torch::Tensor target, float beta, bool size_average){
    torch::Tensor n = torch::abs(input - target);
    auto cond =  n < beta;
    torch::Tensor loss = cond * ((0.5 * n).pow(2) / beta) + (cond == 0) * (n - 0.5 * beta);
    // torch::Tensor loss = torch::where(cond, (0.5 * n).pow(2) / beta, n - 0.5 * beta);
    if(size_average){
        return loss.mean();
    }
    return loss.sum(); 
}