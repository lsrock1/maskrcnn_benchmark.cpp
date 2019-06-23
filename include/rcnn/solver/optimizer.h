#pragma once
#include <torch/torch.h>
#include <torch/ordered_dict.h>
#include <torch/optim/sgd.h>


namespace rcnn{
namespace solver{

class ConcatOptimizer{

public:
  ConcatOptimizer(torch::OrderedDict<std::string, torch::Tensor>& parameters, const double learning_rate, const double momentum, const double weight_decay);
  void zero_grad();
  void step();
  void set_weight_lr(double new_lr);
  void set_bias_lr(double new_lr);
  
  torch::optim::SGD& get_weight_op();
  torch::optim::SGD& get_bias_op();

private:
  torch::optim::SGD weight;
  torch::optim::SGD bias;
};

}
}