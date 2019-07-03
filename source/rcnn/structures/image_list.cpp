#include "image_list.h"
#include <cmath>
#include <iostream>


namespace rcnn{
namespace structures{

ImageList::ImageList(torch::Tensor tensor, std::vector<std::pair<Height, Width>> image_sizes){
  tensors_ = tensor;
  image_sizes_ = image_sizes;
};

ImageList ImageList::to(const torch::Device device){
    return ImageList(tensors_.to(device), image_sizes_);
};

std::vector<std::pair<Height, Width>> ImageList::get_image_sizes() const{
  return image_sizes_;
}

torch::Tensor ImageList::get_tensors() const{
  return tensors_;
}

ImageList ToImageList(torch::Tensor tensors, int size_divisible){
  if(size_divisible > 0){
    std::vector<torch::Tensor> wrapped_tensors{tensors};
    return ToImageList(wrapped_tensors, size_divisible);
  }
  else{
    if(tensors.sizes().size() == 3){
      tensors.unsqueeze_(0);
    }
    std::vector<std::pair<Height, Width>> image_sizes;
    for(int i = 0; i < tensors.size(0); ++i)
      image_sizes.push_back(std::make_pair(tensors[i].size(1), tensors[i].size(2)));
      
    return ImageList(tensors, image_sizes);
  }
}

ImageList ToImageList(ImageList tensors, int size_divisible){
  return tensors;
}

ImageList ToImageList(std::vector<torch::Tensor> tensors, int size_divisible){
  //each tensor dimension in vec == 4
  int64_t max_height = 0;
  int64_t max_width = 0;
  for(int i = 0; i < tensors.size(); ++i){
    if(tensors[i].size(2) > max_height){
      max_height = tensors[i].size(2);
    }

    if(tensors[i].size(3) > max_width){
      max_width = tensors[i].size(3);
    }
  }

  if(size_divisible > 0){
    int64_t stride = size_divisible;
    max_height = static_cast<int64_t>(std::ceil(max_height / static_cast<double>(stride)) * stride);
    max_width = static_cast<int64_t>(std::ceil(max_width / static_cast<double>(stride)) * stride);
  }
  
  torch::Tensor batched_imgs = torch::full({static_cast<int64_t>(tensors.size()), 3, max_height, max_width}, /*fill_value=*/0, torch::TensorOptions().dtype(torch::kF32).device(tensors[0].device()));
  std::vector<std::pair<Height, Width>> image_sizes;
  for(int i = 0; i < tensors.size(); ++i){
    batched_imgs[i].narrow(/*dim=*/1, /*start=*/0, /*length=*/tensors[i].size(2))
                .narrow(/*dim=*/2, /*start=*/0, /*length=*/tensors[i].size(3)).copy_(tensors[i][0]);
    image_sizes.push_back(std::make_pair(tensors[i].size(2), tensors[i].size(3)));
  }
  return ImageList(batched_imgs, image_sizes);
}

}
}