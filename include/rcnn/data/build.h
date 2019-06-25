#pragma once
#include <torch/torch.h>
#include <torch/data/samplers/base.h>
#include "datasets/coco_datasets.h"


namespace rcnn{
namespace data{

//supports only one dataset
//TODO Concat dataset
COCODataset BuildDataset(std::vector<std::string> dataset_list, bool is_train=true);

// template<typename T>
// T MakeDataLoader(bool is_train=true /*, is_distributed=false */, int start_iter=0);


}
}