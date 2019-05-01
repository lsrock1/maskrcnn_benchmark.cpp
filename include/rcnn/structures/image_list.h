#pragma once
#include <torch/torch.h>

namespace rcnn{
namespace structures{
  class ImageList{
    public:
      ImageList(torch::Tensor tensors,  std::pair<int64_t, int64_t> image_sizes);
      ImageList to(const torch::Device device);
    
    private:
      torch::Tensor tensors;
      std::pair<int64_t, int64_t> image_sizes;
  };

  
}
}