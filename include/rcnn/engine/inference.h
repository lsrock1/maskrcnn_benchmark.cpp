#pragma once
#include <map>
#include <bounding_box.h>
#include <modeling.h>

#include <torch/torch.h>


namespace rcnn{
namespace engine{

using namespace rcnn::structures;
using namespace rcnn::modeling;

map<int, BoxList> compute_on_dataset(GeneralizedRCNN model, torch::Device device);
void inference();

}
}