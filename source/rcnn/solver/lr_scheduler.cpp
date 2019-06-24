#include "lr_scheduler.h"
#include "bisect.h"


namespace rcnn{
namespace solver{

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
  return pow(_LRScheduler<torch::optim::SGD>::base_lr * warmup_factor * gamma_, rcnn::utils::bisect_right(milestones_, _LRScheduler<torch::optim::SGD>::last_epoch_)); 
}

ConcatScheduler::ConcatScheduler(ConcatOptimizer& optimizer,
                                std::vector<int64_t> milestones, 
                                float gamma, 
                                float warmup_factor, 
                                int64_t warmup_iters, 
                                std::string warmup_method, 
                                int64_t last_epoch)
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