#pragma once
#include <torch/torch.h>
#include <torch/nn/modules/conv.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

//Reference from https://gist.github.com/mikehamer/df0af5ec7ff98d3cae487975d0c921df
torch::Tensor _NewEmptyTensorOp(const torch::Tensor x, torch::IntArrayRef new_shape);

struct _NewEmptyTensorOpBackward : public torch::autograd::Function{
    torch::IntArrayRef shape;

    torch::autograd::variable_list apply(torch::autograd::variable_list&& grads) override {
        
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
    
    // void release_variables() override {
    //     self_.reset_data();
    //     self_.reset_grad_function();
    // }
};

class eConv2dImpl : public torch::nn::Conv2dImpl{
    public:
        eConv2dImpl(torch::nn::Conv2dOptions conv2dOptions): torch::nn::Conv2dImpl(conv2dOptions){};
        torch::Tensor forward(const Tensor& input);
};

TORCH_MODULE(eConv2d);