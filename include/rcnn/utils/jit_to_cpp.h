#pragma once
#include <torch/torch.h>
#include <torch/script.h>

#include <defaults.h>
#include <modeling.h>


namespace rcnn{
namespace utils{

void recur(std::shared_ptr<torch::jit::script::Module> module, std::string name, std::map<std::string, torch::Tensor>& saved);

void jit_to_cpp(std::string weight_dir, std::string config_path, std::vector<std::string> weight_files);

}
}