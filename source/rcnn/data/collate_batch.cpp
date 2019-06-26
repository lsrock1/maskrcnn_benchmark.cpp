#include "collate_batch.h"
#include <iostream>


namespace rcnn{
namespace data{

BatchCollator::BatchCollator(int size_divisible) :size_divisible_(size_divisible){}

batch BatchCollator::apply_batch(std::vector<torch::data::Example<torch::Tensor, RCNNData>> examples)
{
  std::vector<torch::Tensor> tensors;
  std::vector<rcnn::structures::BoxList> boxes;
  std::vector<int64_t> ids;
  tensors.reserve(examples.size());
  boxes.reserve(examples.size());
  ids.reserve(examples.size());
  for(auto& example : examples){
    tensors.push_back(std::move(example.data));
    boxes.push_back(std::move(example.target.target));
    ids.push_back(std::move(example.target.idx));
  }
  rcnn::structures::ImageList image_list = rcnn::structures::ToImageList(tensors, size_divisible_);

  return std::make_tuple(image_list, boxes, ids);
}

}
}