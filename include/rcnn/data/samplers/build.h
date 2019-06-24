#pragma once
#include "samplers/samplers.h"
#include "datasets/coco_datasets.h"
#include <torch/data/samplers/base.h>


namespace rcnn{
namespace data{

std::shared_ptr<torch::data::samplers::Sampler<>> make_data_sampler(int dataset_size, bool shuffle/*, distributed*/);
std::vector<int> _quantize(std::vector<int> bins, std::vector<int> x);
std::vector<float> _compute_aspect_ratios(COCODataset dataset);
std::shared_ptr<torch::data::samplers::Sampler<>> make_batch_data_sampler(COCODataset dataset, 
                                                                          bool is_train,
                                                                          int start_iter);

}
}