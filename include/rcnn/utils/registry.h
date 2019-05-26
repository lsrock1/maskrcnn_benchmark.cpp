#pragma once
#include <torch/torch.h>
#include <string>


namespace rcnn{
namespace utils{

using backbone = torch::nn::Sequential (*) (void);
backbone BACKBONES(std::string conv_body);


//hard code
}
}

