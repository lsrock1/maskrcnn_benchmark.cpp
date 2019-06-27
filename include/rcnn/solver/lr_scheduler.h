#pragma once
#include <torch/optim/sgd.h>
#include <memory>
#include <cassert>
#include <cmath>

#include "optimizer.h"
#include <iostream>


namespace rcnn{
namespace solver{

template<typename Optimizer>
class _LRScheduler{

public:
  _LRScheduler(Optimizer& optimizer, int last_epoch=-1) :optimizer_(optimizer), base_lr(optimizer.options.learning_rate()){
    if(last_epoch == -1)
      last_epoch_ = 0;
    else
      last_epoch_ = last_epoch;
  }
  //save
  //load
  virtual double get_lr() = 0;
  void step(int epoch = -1){
    if(epoch == -1)
      last_epoch_ += 1;
    else
      last_epoch_ = epoch;
    optimizer_.options.learning_rate(get_lr());
  }

  void set_last_epoch(int64_t last_epoch){
    last_epoch_ = last_epoch;
  }

  void save(torch::serialize::OutputArchive& archive) const{
    archive.write(
      "last_epoch",
      torch::tensor(static_cast<int64_t>(last_epoch_), torch::kI64),
      true
    );
  }

  void load(torch::serialize::InputArchive& archive){
    auto tensor = torch::empty(1, torch::kInt64);
    archive.read(
      "last_epoch",
      tensor,
      true);
    last_epoch_ = tensor.item<int64_t>();
  }

protected:
  Optimizer& optimizer_;
  int64_t last_epoch_ = 0;
  double base_lr;
};


class WarmupMultiStepLR : public _LRScheduler<torch::optim::SGD>{

public:
  WarmupMultiStepLR(torch::optim::SGD& optimizer, 
                    std::vector<int64_t> milestones, 
                    float gamma=0.1, 
                    float warmup_factor=1.0/3, 
                    int64_t warmup_iters=500, 
                    std::string warmup_method="linear", 
                    int64_t last_epoch=-1) 
                    :_LRScheduler<torch::optim::SGD>(optimizer, last_epoch),
                     milestones_(milestones),
                     gamma_(gamma),
                     warmup_factor_(warmup_factor),
                     warmup_iters_(warmup_iters),
                     warmup_method_(warmup_method)
  {assert(warmup_method.compare("linear") == 0 || warmup_method.compare("constant") == 0); step(0);}

  double get_lr() override;

private:
  std::vector<int64_t> milestones_;
  float gamma_;
  float warmup_factor_;
  int64_t warmup_iters_;
  std::string warmup_method_;

};

class ConcatScheduler{

public:
  ConcatScheduler(ConcatOptimizer& optimizer,
                  std::vector<int64_t> milestones, 
                  float gamma=0.1, 
                  float warmup_factor=1.0/3, 
                  int64_t warmup_iters=500, 
                  std::string warmup_method="linear", 
                  int64_t last_epoch=-1);
  void step();
  void load(torch::serialize::InputArchive& archive);
  void save(torch::serialize::OutputArchive& archive) const;
  void set_last_epoch(int64_t last_epoch);

private:
  std::shared_ptr<_LRScheduler<torch::optim::SGD>> weight;
  std::shared_ptr<_LRScheduler<torch::optim::SGD>> bias;
};

}
}