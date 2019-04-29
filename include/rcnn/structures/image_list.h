#pragma once

#include <torch/torch.h>

namespace mrcn{
namespace structures{
  class ImageList{
    public:
      ImageList(torch::Tensor tensors, vector);  
  }
}
}