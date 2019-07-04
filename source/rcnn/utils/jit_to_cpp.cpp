#include "jit_to_cpp.h"

#include <iostream>
#include <cassert>

#include <torch/script.h>

#include <defaults.h>
#include <modeling.h>


namespace rcnn{
namespace utils{

void recur(std::shared_ptr<torch::jit::script::Module> module, std::string name, std::map<std::string, torch::Tensor>& saved){
  std::string new_name;
  if(name.compare("") != 0)
    new_name = name + ".";
  
  for(auto& u : module->get_parameters()){
    torch::Tensor tensor = u.value().toTensor();
    saved[new_name + u.name()] = tensor;
  }
  for(auto& u : module->get_attributes()){
    torch::Tensor tensor = u.value().toTensor();
    saved[new_name + u.name()] = tensor;
  }
  for(auto& i : module->get_modules())
    recur(i, new_name + i->name(), saved);
}

void jit_to_cpp(std::string weight_dir, std::string config_path, std::vector<std::string> weight_files)
{
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

  // for(auto i = saved.begin(); i != saved.end(); ++i)
  //   std::cout << i->first << "\n";

  for(auto& i : model->named_parameters()){
    std::string name;
    if(i.key().find("backbone") != std::string::npos){
      name = i.key().substr(20);
      if(i.key().find("fpn") != std::string::npos){
        name = name.substr(0, 14);
        if(i.key().find("weight") !=std::string::npos){
          name += ".weight";
        }
        else{
          name += ".bias";
        }
      }

      for(auto s = saved.begin(); s != saved.end(); ++s){
        if((s->first).find(name) != std::string::npos){
          i.value().copy_(s->second);
          updated.insert(i.key());
          mapping[i.key()] = s->first;
        }
      }
    }
    else if(i.key().find("rpn") != std::string::npos){
      if(i.key().find("conv") != std::string::npos){
        if(i.key().find("weight") != std::string::npos)
          name = "rpn_conv.weight";
        else
          name = "rpn_conv.bias";
        i.value().copy_(saved.at(name));
        updated.insert(i.key());
        mapping[i.key()] = name;
      }
      else if(i.key().find("bbox") != std::string::npos){
        if(i.key().find("weight") != std::string::npos)
          name = "rpn_bbox.weight";
        else
          name = "rpn_bbox.bias";
        i.value().copy_(saved.at(name));
        updated.insert(i.key());
        mapping[i.key()] = name;
      }
      else if(i.key().find("logits") != std::string::npos){
        if(i.key().find("weight") != std::string::npos)
          name = "rpn_logits.weight";
        else
          name = "rpn_logits.bias";
        i.value().copy_(saved.at(name));
        updated.insert(i.key());
        mapping[i.key()] = name;
      }
      else{
        assert(false);
      }
    }
    else if(i.key().find("roi_heads.box") != std::string::npos){
      if(i.key().find("cls_score") != std::string::npos){
        if(i.key().find("weight") != std::string::npos)
          name = "cls_score.weight";
        else
          name = "cls_score.bias";
        i.value().copy_(saved.at(name));
        updated.insert(i.key());
        mapping[i.key()] = name;
      }
      else if(i.key().find("bbox_pred") != std::string::npos){
        if(i.key().find("weight") != std::string::npos)
          name = "bbox_pred.weight";
        else
          name = "bbox_pred.bias";
        i.value().copy_(saved.at(name));
        updated.insert(i.key());
        mapping[i.key()] = name;
      }
      else if(i.key().find(".head.") != std::string::npos){
        name = i.key().substr(39);
        for(auto s = saved.begin(); s != saved.end(); ++s){
          if((s->first).find(name) != std::string::npos){
            i.value().copy_(s->second);
            updated.insert(i.key());
            mapping[i.key()] = s->first;
          }
        }
      }
      else if(i.key().find("fc") != std::string::npos){
        name = "extractor_" + i.key().substr(i.key().find("fc"));
        i.value().copy_(saved.at(name));
        updated.insert(i.key());
        mapping[i.key()] = name;
      }
      else{
        assert(false);
      }
    }
    else{
      assert(false);
    }
  }

  for(auto& i : model->named_buffers()){
    std::string name;
    if(i.key().find("backbone") != std::string::npos){
      name = i.key().substr(20);
      for(auto s = saved.begin(); s != saved.end(); ++s){
        if((s->first).find(name) != std::string::npos){
          i.value().copy_(s->second);
          updated.insert(i.key());
          mapping[i.key()] = s->first;
        }
      }
    }
    else if(i.key().find("roi_heads.box") != std::string::npos){
      if(i.key().find("head") != std::string::npos){
        name = i.key().substr(39);
        for(auto s = saved.begin(); s != saved.end(); ++s){
          if((s->first).find(name) != std::string::npos){
            i.value().copy_(s->second);
            updated.insert(i.key());
            mapping[i.key()] = s->first;
          }
        }
      }
      else{
        assert(false);
      }
    }
    else if(i.key().find("anchor") != std::string::npos){
      updated.insert(i.key());
       mapping[i.key()] = i.key();
    }
    else{
      assert(false);
    }
  }

  //check
  for(auto& i : model->named_parameters()){
    assert(updated.count(i.key()));
    std::cout << i.key() << "load from " << mapping[i.key()] << "\n";
  }

  //check
  for(auto& i : model->named_buffers()){
    assert(updated.count(i.key()));
    std::cout << i.key() << "load from " << mapping[i.key()] << "\n";
  }

  torch::serialize::OutputArchive archive;
  for(auto& i : model->named_parameters()){
    assert(updated.count(i.key()));
    assert( (saved.at(mapping[i.key()]) != i.value()).sum().item<int>() == 0);
    archive.write(i.key(), i.value());
  }

  for(auto& i : model->named_buffers()){
    assert(updated.count(i.key()));
    if(i.key().find("anchor_generator") == std::string::npos)
      assert( (saved.at(mapping[i.key()]) != i.value()).sum().item<int>() == 0);
    archive.write(i.key(), i.value(), true);
  }
  archive.save_to("../models/new_pth_from_python_cpp.pth");
  std::cout << "saved as /models/new_pth_from_python_cpp.pth\n"; 

}
}
}