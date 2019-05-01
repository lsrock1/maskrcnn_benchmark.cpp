#include "image_list.h"

namespace rcnn{
namespace structures{
ImageList::ImageList(torch::Tensor tensor, std::pair<int64_t, int64_t> image_sizes){
  this->tensors = tensor;
  this->image_sizes = image_sizes;
};

ImageList ImageList::to(const torch::Device device){
    return ImageList(this->tensors.to(device), this->image_sizes);
};
}
}