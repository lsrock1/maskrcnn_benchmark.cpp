#include "optimizer.h"


namespace rcnn{
namespace solver{

ConcatOptimizer::ConcatOptimizer(torch::OrderedDict<std::string, torch::Tensor>& parameters, 
                                const double learning_rate, 
                                const double momentum, 
                                const double weight_decay)
{
  std::vector<torch::Tensor> bias_tensor;
  std::vector<torch::Tensor> weight_tensor;
  for(auto& param : parameters){
    if(param.key.compare("bias") != std::string::npos)
      bias_tensor.push_back(param.value);
    else
      weight_tensor.push_back(param.value);
  }
  weight = torch::optim::SGD(weight_tensor, torch::optim::SGDOptions(learning_rate).momentum(momentum).weight_decay(weight_decay));
  bias = torch::optim::SGD(bias_tensor, torch::optim::SGDOptions(learning_rate).momentum(momentum).weight_decay(weight_decay));
}

void ConcatOptimizer::zero_grad(){
  weight.zero_grad();
  bias.zero_grad();
}

void ConcatOptimizer::step(){
  weight.step();
  bias.step();
}

void ConcatOptimizer::set_weight_lr(double new_lr){
  weight.options.learning_rate = new_lr;
}
void ConcatOptimizer::set_bias_lr(double new_lr){
  bias.options.learning_rate = new_lr;
}

torch::optim::SGD& ConcatOptimizer::get_weight_op(){
  return weight;
}

torch::optim::SGD& ConcatOptimizer::get_bias_op(){
  return bias;
}

}
}