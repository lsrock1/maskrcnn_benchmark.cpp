#pragma once
#include <torch/torch.h>
#include <torch/data/example.h>
#include <torch/data/transforms/tensor.h>

#include "datasets/coco_datasets.h"


namespace rcnn{
namespace data{

template<typename InputType, typename OutputType>
class RCNNTransform : public torch::data::transforms::Transform<torch::data::Example<InputType, RCNNData>, torch::data::Example<OutputType, RCNNData>>{
  
public:  
  virtual torch::data::Example<OutputType, RCNNData> operator()(torch::data::Example<InputType, RCNNData> input) = 0;

  torch::data::Example<OutputType, RCNNData> apply(torch::data::Example<InputType, RCNNData> input) override{
    return (*this)(input);
  }
};

using MatToTensorTransform = RCNNTransform<cv::Mat, torch::Tensor>;
using MatToMatTransform = RCNNTransform<cv::Mat, cv::Mat>;
using TensorToTensorTransform = RCNNTransform<torch::Tensor, torch::Tensor>;


class Resize : public MatToMatTransform{

public:
  Resize(int min_size, int max_size):min_size_(min_size), max_size_(max_size){};
  std::pair<int, int> get_size(std::pair<int, int> image_size);
  torch::data::Example<cv::Mat, RCNNData> operator()(torch::data::Example<cv::Mat, RCNNData> input) override;

private:
  int min_size_;
  int max_size_;
};

class RandomHorizontalFlip : public MatToMatTransform{

public:
  RandomHorizontalFlip(float prob = 0.5):prob_(prob){};
  torch::data::Example<cv::Mat, RCNNData> operator()(torch::data::Example<cv::Mat, RCNNData> input) override;

private:
  float prob_;
};

class RandomVerticalFlip : public MatToMatTransform{

public:
  RandomVerticalFlip(float prob = 0.5):prob_(prob){};
  torch::data::Example<cv::Mat, RCNNData> operator()(torch::data::Example<cv::Mat, RCNNData> input) override;

private:
  float prob_;
};

class ToTensor : public MatToTensorTransform{

public:
  torch::data::Example<torch::Tensor, RCNNData> operator()(torch::data::Example<cv::Mat, RCNNData> input) override;
};

class Normalize : public TensorToTensorTransform{

public:
  Normalize(torch::ArrayRef<double> mean, torch::ArrayRef<double> stddev, bool to_bgr255);
  torch::data::Example<torch::Tensor, RCNNData> operator()(torch::data::Example<torch::Tensor, RCNNData> input) override;

private:
  torch::Tensor mean, stddev;
  bool to_bgr255_;
};

class Compose : public MatToTensorTransform{

public:
  Compose(std::vector<MatToMatTransform*> MtoMtransforms, std::vector<TensorToTensorTransform*> TtoTtransforms);
  torch::data::Example<torch::Tensor, RCNNData> operator()(torch::data::Example<cv::Mat, RCNNData> input) override;

private:
  std::vector<MatToMatTransform*> MtoMtransforms_;
  ToTensor to_tensor;
  std::vector<TensorToTensorTransform*> TtoTtransforms_;
};

}//data
}//rcnn