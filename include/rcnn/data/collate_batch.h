#pragma once

#include <torch/data/example.h>
#include <torch/data/transforms/collate.h>
#include <bounding_box.h>
#include "datasets/coco_datasets.h"


namespace rcnn{
namespace data{

using batch = std::tuple<std::vector<torch::Tensor>, std::vector<rcnn::structures::BoxList>, std::vector<int64_t>>;
//<output, input>
struct BatchCollator : public torch::data::transforms::Collation<batch, std::vector<torch::data::Example<torch::Tensor, RCNNData>>>{

  batch apply_batch(std::vector<torch::data::Example<torch::Tensor, RCNNData>> examples) override;

};

}
}