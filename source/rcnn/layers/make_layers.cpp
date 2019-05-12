#include "make_layers.h"
#include "conv2d.h"

namespace rcnn{
namespace layers{
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
}
}