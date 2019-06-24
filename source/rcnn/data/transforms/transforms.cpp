#include "transforms/transforms.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>


namespace rcnn{
namespace data{

Compose::Compose(std::vector<MatToMatTransform*> MtoMtransforms,
                 std::vector<TensorToTensorTransform*> TtoTtransforms)
                 :MtoMtransforms_(MtoMtransforms),
                  to_tensor(),
                  TtoTtransforms_(TtoTtransforms){
                    std::srand(static_cast<unsigned int>(std::time(0)));
                  }

torch::data::Example<torch::Tensor, RCNNData> Compose::operator()(torch::data::Example<cv::Mat, RCNNData> input){
  torch::data::Example<torch::Tensor, RCNNData> tensor_rcnn;
  bool tensor_init = false;

  for(auto& MtoM : MtoMtransforms_)
    input = (*MtoM)(input);
  
  tensor_rcnn = to_tensor(input);

  for(auto& TtoT : TtoTtransforms_)
    tensor_rcnn = (*TtoT)(tensor_rcnn);

  return tensor_rcnn;    
}

std::pair<int, int> Resize::get_size(std::pair<int, int> image_size){
  int w, h, size, ow, oh;
  std::tie(w, h) = image_size;
  size = min_size_;
  float min_original_size = static_cast<float>(std::min(w, h));
  float max_original_size = static_cast<float>(std::max(w, h));

  if(max_original_size / min_original_size * size > max_size_)
    size = static_cast<int>(std::round(max_size_ * min_original_size / max_original_size));

  if((w <= h && w == size) || (h <= w && h == size))
    return std::make_pair(h, w);

  if(w < h){
    ow = size;
    oh = size * h / w;
  }
  else{
    oh = size;
    ow = size * w / h;
  }
  return std::make_pair(oh, ow);
}

torch::data::Example<cv::Mat, RCNNData> Resize::operator()(torch::data::Example<cv::Mat, RCNNData> input){
  int h, w;
  cv::Mat resized;
  std::tie(h, w) = get_size(std::make_pair(input.data.cols, input.data.rows));
  cv::resize(input.data, resized, cv::Size(w, h));
  input.data = resized;
  input.target.target = input.target.target.Resize(std::make_pair(w, h));
  return input;
}

torch::data::Example<cv::Mat, RCNNData> RandomHorizontalFlip::operator()(torch::data::Example<cv::Mat, RCNNData> input){
  float r = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
  if(r < prob_){
    cv::Mat flipped;
    cv::flip(input.data, flipped, 1);
    input.data = flipped;
    input.target.target = input.target.target.Transpose(structures::FLIP_LEFT_RIGHT);
  }
  return input;
}

torch::data::Example<cv::Mat, RCNNData> RandomVerticalFlip::operator()(torch::data::Example<cv::Mat, RCNNData> input){
  float r = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
  if(r < prob_){
    cv::Mat flipped;
    cv::flip(input.data, flipped, 0);
    input.data = flipped;
    input.target.target = input.target.target.Transpose(structures::FLIP_TOP_BOTTOM);
  }
  return input;
}

torch::data::Example<torch::Tensor, RCNNData> ToTensor::operator()(torch::data::Example<cv::Mat, RCNNData> input){
  torch::Tensor tensor_image = torch::from_blob(input.data.data, {1, 3, input.data.rows, input.data.cols}, torch::kByte);
  tensor_image = tensor_image.to(torch::kFloat);
  return torch::data::Example<torch::Tensor, RCNNData> {tensor_image, input.target};
}

Normalize::Normalize(torch::ArrayRef<double> mean, torch::ArrayRef<double> stddev, bool to_bgr255)
          : mean(torch::tensor(mean, torch::kFloat32)
                .unsqueeze(/*dim=*/1)
                .unsqueeze(/*dim=*/2)),
            stddev(torch::tensor(stddev, torch::kFloat32)
                .unsqueeze(/*dim=*/1)
                .unsqueeze(/*dim=*/2)),
            to_bgr255_(to_bgr255) {}

torch::data::Example<torch::Tensor, RCNNData> Normalize::operator()(torch::data::Example<torch::Tensor, RCNNData> input){
  if(!to_bgr255_)
    input.data.div_(255);
  input.data = input.data.sub(mean).div(stddev);
  return input;
}


}//data
}//rcnn