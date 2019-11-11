#pragma once
#include <torch/torch.h>

namespace rcnn {
namespace structures {

namespace {
  using Width = int64_t;
  using Height = int64_t;
}

class ImageList {
  
public:
  ImageList(torch::Tensor tensors,  std::vector<std::pair<Height, Width>> image_sizes);
  ImageList to(const torch::Device device);
  std::vector<std::pair<Height, Width>> get_image_sizes() const;
  torch::Tensor get_tensors() const;
  
private:
  torch::Tensor tensors_;
  std::vector<std::pair<Height, Width>> image_sizes_;
};

ImageList ToImageList(torch::Tensor tensors, int size_divisible = 0);
ImageList ToImageList(ImageList tensors, int size_divisible = 0);
ImageList ToImageList(std::vector<torch::Tensor> tensors, int size_divisible = 0);

} // namespace structures
} // namespace rcnn
