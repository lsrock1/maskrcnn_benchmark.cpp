#include "misc.h"
#include <torch/torch.h>

namespace rcnn{
namespace layers{

torch::autograd::variable_list _NewEmptyTensorOpBackward::apply(torch::autograd::variable_list&& grads) {
    // Our function had one output, so we only expect 1 gradient
    auto& grad = grads[0];

    // Variable list to hold the gradients at the function's input variables
    torch::autograd::variable_list grad_inputs(1); 

    // Do gradient computation for each of the inputs
    if (should_compute_output(0)) {
        auto grad_result = _NewEmptyTensorOp(grad, shape);
        grad_inputs[0] = grad_result;
    }

    return grad_inputs;
}


torch::Tensor Conv2dImpl::forward(const torch::Tensor& input){
    if(input.numel() > 0){
        return torch::nn::Conv2dImpl::forward(input);
    }
    int64_t height = (input.size(2) + 2 * options.padding_.size() - options.dilation_.size() * (options.kernel_size_.size() - 1) - 1) / options.stride_.size() + 1;
    int64_t width = (input.size(3) + 2 * options.padding_.size() - options.dilation_.size() * (options.kernel_size_.size() - 1) - 1) / options.stride_.size() + 1;
    torch::IntArrayRef shape = torch::IntArrayRef({input.size(0), weight.size(0), height, width});

    return _NewEmptyTensorOp(input, shape);
};


torch::Tensor _NewEmptyTensorOp(const torch::Tensor x, torch::IntArrayRef new_shape){
  auto& self_ = torch::autograd::as_variable_ref(x);
  auto result = torch::empty(new_shape, torch::TensorOptions().dtype(self_.dtype()).requires_grad(self_.requires_grad()));
  // auto tmp = self_.data().new_empty(new_shape);
  //auto result = torch::autograd::as_variable(tmp);

  if(x.requires_grad()){
    auto grad_fn = std::shared_ptr<_NewEmptyTensorOpBackward>(new _NewEmptyTensorOpBackward(), torch::autograd::deleteFunction);
    grad_fn -> set_next_edges(torch::autograd::collect_next_edges(x));
    grad_fn -> shape = x.sizes();
    set_history(torch::autograd::flatten_tensor_args( result ), grad_fn);
  }
  return result;
};

torch::nn::Sequential ConvWithKaimingUniform(/*NO GN use_gn=false, */bool use_relu, int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t dilation){
  torch::nn::Sequential module;
  module->push_back(Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                        .padding(dilation * static_cast<int>((kernel_size - 1) / 2))
                                        .stride(stride)
                                        .with_bias(true)));
  if(use_relu){
    module->push_back(torch::nn::Functional(torch::relu));
  }
  for(auto &param : module->named_parameters()){
    if(param.key().find("weight") != std::string::npos) {
      torch::nn::init::kaiming_uniform_(param.value(), 1);
    }
    else if(param.key().find("bias") != std::string::npos){
      torch::nn::init::zeros_(param.value());
    }
  }
  return module;
}

torch::nn::Linear MakeFC(int64_t dim_in, int64_t hidden_dim/*, use_gn*/){
  torch::nn::Linear fc = torch::nn::Linear(dim_in, hidden_dim);
  for(auto &param : fc->named_parameters()){
    if(param.key().find("weight") != std::string::npos) {
      torch::nn::init::kaiming_uniform_(param.value(), 1);
    }
    else if(param.key().find("bias") != std::string::npos){
      torch::nn::init::zeros_(param.value());
    }
  }
  return fc;
}

torch::nn::Sequential MakeConv3x3(int64_t in_channels, int64_t out_channels, int64_t dilation, int64_t stride, /*use_gn, */bool use_relu, bool kaiming_init){
  torch::nn::Sequential module;
  module->push_back(Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                                        .padding(dilation)
                                        .stride(stride)
                                        .with_bias(true)));
  if(kaiming_init){
    for(auto &param : module->named_parameters()){
      if(param.key().find("weight") != std::string::npos) {
        torch::nn::init::kaiming_uniform_(param.value(), 1);
      }
      else if(param.key().find("bias") != std::string::npos){
        torch::nn::init::zeros_(param.value());
      }
    }
  }
  else{
    for(auto &param : module->named_parameters()){
      if(param.key().find("weight") != std::string::npos) {
        torch::nn::init::normal_(param.value(), 0.01);
      }
      else if(param.key().find("bias") != std::string::npos){
        torch::nn::init::zeros_(param.value());
      }
    } 
  }

  if(use_relu){
      module->push_back(torch::nn::Functional(torch::relu));
  }
  return module;
}

}//layers
}//rcnn