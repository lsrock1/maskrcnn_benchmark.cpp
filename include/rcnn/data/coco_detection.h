#pragma once
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "bounding_box.h"
#include "coco.h"


namespace rcnn{
namespace data{

class COCODetection : public torch::data::datasets::Dataset<COCODetection, torch::data::Example<cv::Mat, std::vector<coco::Annotation>>> {

public:
  COCODetection(std::string root, std::string annFile/*TODO transform=*/);
  torch::data::Example<cv::Mat, std::vector<coco::Annotation>> get(size_t index) override;
  torch::optional<size_t> size() const override;

  std::string root_;
  coco::COCO coco_;
  std::vector<int> ids_;

friend std::ostream& operator << (std::ostream& os, const COCODetection& bml);
};

}//data
}//rcnn

