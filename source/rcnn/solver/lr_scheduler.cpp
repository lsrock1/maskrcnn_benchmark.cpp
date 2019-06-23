#include "lr_scheduler.h"


namespace rcnn{
namespace solver{

double bisect_right(std::vector<int> milestones, int element){
  int index = 0;
  for(auto& i : milestones){
    if(element < i)
      return index;
    else if(element == i)
      return index + 1;
    else
      index += 1;
  }
  return static_cast<double>(index);
}

double WarmupMultiStepLR::get_lr(){
  double warmup_factor = 1;
  if(_LRScheduler<torch::optim::SGD>::last_epoch_ < warmup_iters_){
    if(warmup_method_.compare("constant") == 0)
      warmup_factor = warmup_factor_;
    else if(warmup_method_.compare("linear") == 0){
      double alpha = static_cast<double>(_LRScheduler<torch::optim::SGD>::last_epoch_) / warmup_iters_;
      warmup_factor = warmup_factor_ * (1. - alpha) + alpha;
    }
  }
  return pow(_LRScheduler<torch::optim::SGD>::base_lr * warmup_factor * gamma_, bisect_right(milestones_, _LRScheduler<torch::optim::SGD>::last_epoch_)); 
}

ConcatScheduler::ConcatScheduler(ConcatOptimizer& optimizer,
                                std::vector<int> milestones, 
                                float gamma, 
                                double warmup_factor, 
                                int warmup_iters, 
                                std::string warmup_method, 
                                int last_epoch)
                                :weight(
                                  WarmupMultiStepLR(
                                    optimizer.get_weight_op(),
                                    milestones,
                                    gamma,
                                    warmup_factor,
                                    warmup_iters,
                                    warmup_method,
                                    last_epoch
                                )),
                                 bias(
                                   WarmupMultiStepLR(
                                    optimizer.get_bias_op(),
                                    milestones,
                                    gamma,
                                    warmup_factor,
                                    warmup_iters,
                                    warmup_method,
                                    last_epoch
                                )){}

void ConcatScheduler::step(){
  weight.step();
  bias.step();
}

}
}