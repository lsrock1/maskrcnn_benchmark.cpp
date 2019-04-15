#include <layers/misc.h>
#include <torch/torch.h>

torch::Tensor eConv2dImpl::forward(const torch::Tensor& input){
    if(input.numel() > 0){
        return torch::nn::Conv2dImpl::forward(input);
    }
    int64_t height = (input.size(2) + 2 * options.padding_.size() - (options.dilation_.size() * (options.kernel_size_.size() - 1) - 1)) / options.stride_.size() + 1;
    int64_t width = (input.size(3) + 2 * options.padding_.size() - (options.dilation_.size() * (options.kernel_size_.size() - 1) - 1)) / options.stride_.size() + 1;
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
}