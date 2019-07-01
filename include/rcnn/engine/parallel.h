#pragma once
#include <torch/torch.h>
#ifdef USE_CUDA
#include <torch/nn/parallel/data_parallel.h>
#endif
#include <modeling.h>
#include <image_list.h>
#include <bounding_box.h>


namespace rcnn{
namespace engine{

std::vector<std::map<std::string, torch::Tensor>> parallel_apply(
    std::vector<rcnn::modeling::GeneralizedRCNN>& modules,
    const std::vector<rcnn::structures::ImageList>& inputs,
    const std::vector<std::vector<rcnn::structures::BoxList>>& targets,
    const torch::optional<std::vector<torch::Device>>& devices = torch::nullopt);

std::pair<torch::Tensor, std::map<std::string, torch::Tensor>> data_parallel(
    rcnn::modeling::GeneralizedRCNN module,
    rcnn::structures::ImageList images, 
    std::vector<rcnn::structures::BoxList> targets,
    torch::optional<std::vector<torch::Device>> devices = torch::nullopt,
    torch::optional<torch::Device> output_device = torch::nullopt,
    int64_t dim = 0);

}
}