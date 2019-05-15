//Reference from https://gist.github.com/mikehamer/df0af5ec7ff98d3cae487975d0c921df
#include "conv2d.h"


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
  auto result = torch::empty(new_shape, torch::TensorOptions().dtype(self_.dtype()).device(self_.device()));
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

}//layers
}//rcnn