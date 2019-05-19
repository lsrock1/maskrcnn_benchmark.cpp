#pragma once
#include <torch/torch.h>


namespace rcnn{
namespace modeling{

torch::Tensor PermuteAndFlatten(torch::Tensor layer, int64_t N, int64_t A, int64_t C, int64_t H, int64_t W);

}
}