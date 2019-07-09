#pragma once
#include <torch/torch.h>
#include <torch/script.h>

#include <defaults.h>
#include <modeling.h>

#include <iostream>
#include <bisect.h>


namespace rcnn{
namespace utils{

void recur(std::shared_ptr<torch::jit::script::Module> module, std::string name, std::map<std::string, torch::Tensor>& saved);

template<typename T>
void jit_to_cpp(std::string weight_dir, std::string config_path, std::vector<std::string> weight_files){
  T mapper = T();
  std::map<std::string, torch::Tensor> saved;
  std::set<std::string> updated;
  std::map<std::string, std::string> mapping;

  rcnn::config::SetCFGFromFile(config_path);
  modeling::GeneralizedRCNN model = modeling::BuildDetectionModel();
  torch::NoGradGuard guard;

  for(auto& weight_file : weight_files){
    auto module_part = torch::jit::load(weight_dir + "/" + weight_file);
    recur(module_part, weight_file.substr(0, weight_file.size()-4), saved);
  }

  for(auto& i : model->named_parameters()){
    std::string new_name;
    if(i.key().find("backbone") != std::string::npos){
      new_name = mapper.backboneMapping(i.key(), i.value(), saved);
      updated.insert(i.key());
      mapping[i.key()] = new_name;
    }
    else if(i.key().find("rpn") != std::string::npos){
      new_name = mapper.rpn(i.key(), i.value(), saved);
      updated.insert(i.key());
      mapping[i.key()] = new_name;
    }
    else if(i.key().find("roi_heads.box") != std::string::npos){
      new_name = mapper.roiHead(i.key(), i.value(), saved);
      updated.insert(i.key());
      mapping[i.key()] = new_name;
    }
    else{
      assert(false);
    }
  }
  
  for(auto& i : model->named_buffers()){
    std::string new_name;
    if(i.key().find("backbone") != std::string::npos){
      new_name = mapper.backboneMapping(i.key(), i.value(), saved);
      updated.insert(i.key());
      mapping[i.key()] = new_name;
    }
    else if(i.key().find("rpn") != std::string::npos){
      new_name = mapper.rpn(i.key(), i.value(), saved);
      updated.insert(i.key());
      mapping[i.key()] = new_name;
    }
    else if(i.key().find("roi_heads.box") != std::string::npos){
      new_name = mapper.roiHead(i.key(), i.value(), saved);
      updated.insert(i.key());
      mapping[i.key()] = new_name;
    }
    else{
      assert(false);
    }
  }

  torch::serialize::OutputArchive archive;
  for(auto& i : model->named_parameters()){
    std::cout << i.key() << " parameter loaded from " << mapping[i.key()] << "\n";
    assert(updated.count(i.key()));
    assert((saved.at(mapping[i.key()]) != i.value()).sum().item<int>() == 0);
    archive.write(i.key(), i.value());
  }

  for(auto& i : model->named_buffers()){
      std::cout << i.key() << " buffer loaded from " << mapping[i.key()] << "\n";
    assert(updated.count(i.key()));
    if(i.key().find("anchor_generator") == std::string::npos){
      assert( (saved.at(mapping[i.key()]) != i.value()).sum().item<int>() == 0);
    }
    archive.write(i.key(), i.value(), true);
  }
  archive.save_to("../models/new_pth_from_python_cpp.pth");
  std::cout << "saved as /models/new_pth_from_python_cpp.pth\n"; 
}

class ResNetMapper{

public:
  ResNetMapper() = default;
  std::string backboneMapping(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved);
  std::string roiHead(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved);
  std::string rpn(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved);

};

class VoVNetMapper{

public:
  VoVNetMapper() = default;
  std::string backboneMapping(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved);
  std::string roiHead(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved);
  std::string rpn(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved);
};


}
}