#pragma once
#include <torch/torch.h>

namespace rcnn{
namespace engine{

void do_train(int checkpoint_period, int iteration , torch::Device device = torch::Device("gpu"));

}
}