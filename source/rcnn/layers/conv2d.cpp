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
  int64_t stride = static_cast<torch::ArrayRef<int64_t>>(options.stride_).at(0),
          padding = static_cast<torch::ArrayRef<int64_t>>(options.padding_).at(0), 
          dilation = static_cast<torch::ArrayRef<int64_t>>(options.dilation_).at(0), 
          output_padding = static_cast<torch::ArrayRef<int64_t>>(options.output_padding_).at(0),
          kernel_size = static_cast<torch::ArrayRef<int64_t>>(options.kernel_size_).at(0);
  torch::IntArrayRef shape;
  if(options.transposed()){
    int64_t height = (input.size(2) - 1) * stride - 2 * padding + (dilation * (kernel_size - 1) + 1) + output_padding;
    int64_t width = (input.size(3) - 1) * stride - 2 * padding + (dilation * (kernel_size - 1) + 1) + output_padding;
    shape = torch::IntArrayRef({input.size(0), bias.size(0), height, width});
  }
  else{
    int64_t height = (input.size(2) + 2 *  padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    int64_t width = (input.size(3) + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1;
    shape = torch::IntArrayRef({input.size(0), weight.size(0), height, width});
  }
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
}

}//layers
}//rcnn