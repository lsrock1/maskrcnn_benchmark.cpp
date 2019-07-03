#include "collate_batch.h"
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"


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
    // cv::Mat warp = cv::Mat(example.data.size(2), example.data.size(3), CV_32FC3, example.data.permute({0, 2, 3, 1}).contiguous().data<float>());
    // cv::imwrite("../resource/tmp/collate" + std::to_string(example.target.idx) + ".jpg", warp);
    tensors.push_back(example.data);
    boxes.push_back(example.target.target);
    ids.push_back(example.target.idx);
  }
  rcnn::structures::ImageList image_list = rcnn::structures::ToImageList(tensors, size_divisible_);
  // for(int i = 0; i < tensors.size(); ++i){
  //   cv::Mat warpt = cv::Mat(image_list.get_tensors()[i].size(1), image_list.get_tensors()[i].size(2), CV_32FC3, image_list.get_tensors()[i].permute({1, 2, 0}).contiguous().data<float>());
  //   cv::imwrite("../resource/tmp/divisible" + std::to_string(i) + ".jpg", warpt);
  // }
  return std::make_tuple(image_list, boxes, ids);
}

}
}