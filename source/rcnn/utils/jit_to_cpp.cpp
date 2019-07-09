#include "jit_to_cpp.h"

#include <cassert>
#include <iostream>


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

std::string ResNetMapper::backboneMapping(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved){
  std::string new_name = name.substr(20);
  if(name.find("fpn") != std::string::npos){
    new_name = new_name.substr(0, 14);
    if(name.find("weight") != std::string::npos){
      new_name += ".weight";
    }
    else{
      new_name += ".bias";
    }
  }

  for(auto s = saved.begin(); s != saved.end(); ++s){
    if((s->first).find(new_name) != std::string::npos){
      value.copy_(s->second);
      return s->first;
      // updated.insert(i.key());
      // mapping[i.key()] = s->first;
    }
  }
  assert(false);
}

std::string ResNetMapper::roiHead(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved){
  std::string new_name;
  if(name.find("cls_score") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "cls_score.weight";
    else
      new_name = "cls_score.bias";
    value.copy_(saved.at(new_name));
    return new_name;
  }
  else if(name.find("bbox_pred") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "bbox_pred.weight";
    else
      new_name = "bbox_pred.bias";
    value.copy_(saved.at(new_name));
    // updated.insert(i.key());
    // mapping[i.key()] = name;
    return new_name;
  }
  else if(name.find(".head.") != std::string::npos){
    new_name = name.substr(39);
    for(auto s = saved.begin(); s != saved.end(); ++s){
      if((s->first).find(new_name) != std::string::npos){
        value.copy_(s->second);
        // updated.insert(i.key());
        // mapping[i.key()] = s->first;
        return s->first;
      }
    }
    assert(false);
  }
  else if(name.find("fc") != std::string::npos){
    new_name = "extractor_" + name.substr(name.find("fc"));
    value.copy_(saved.at(new_name));
    // updated.insert(i.key());
    // mapping[i.key()] = name;
    return new_name;
  }
  else{
    assert(false);
  }
}

std::string ResNetMapper::rpn(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved){
  std::string new_name;
  if(name.find("conv") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "rpn_conv.weight";
    else
      new_name = "rpn_conv.bias";
    value.copy_(saved.at(new_name));
    return new_name;
  }
  else if(name.find("bbox") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "rpn_bbox.weight";
    else
      new_name = "rpn_bbox.bias";
    value.copy_(saved.at(new_name));
    // updated.insert(i.key());
    // mapping[i.key()] = name;
    return new_name;
  }
  else if(name.find("logits") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "rpn_logits.weight";
    else
      new_name = "rpn_logits.bias";
    value.copy_(saved.at(new_name));
    return new_name;
  }
  else{
    assert(false);
  }
}

std::string VoVNetMapper::backboneMapping(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved){
  
  std::string new_name = name.substr(20);
  if(name.find("fpn") != std::string::npos){
    new_name = new_name.substr(0, 14);
    if(name.find("weight") != std::string::npos){
      new_name += ".weight";
    }
    else{
      new_name += ".bias";
    }
  }
  else if(name.find("stem") == std::string::npos){
    new_name = new_name.substr(13);
    if(name.find("layers_") != std::string::npos){
      new_name.replace(new_name.find("layers_") + 6, 1, ".");
    }
  }

  for(auto s = saved.begin(); s != saved.end(); ++s){
    if((s->first).find(new_name) != std::string::npos){
      value.copy_(s->second);
      return s->first;
      // updated.insert(i.key());
      // mapping[i.key()] = s->first;
    }
  }
  assert(false);
}

std::string VoVNetMapper::roiHead(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved){
  std::string new_name;
  if(name.find("cls_score") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "cls_score.weight";
    else
      new_name = "cls_score.bias";
    value.copy_(saved.at(new_name));
    return new_name;
  }
  else if(name.find("bbox_pred") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "bbox_pred.weight";
    else
      new_name = "bbox_pred.bias";
    value.copy_(saved.at(new_name));
    return new_name;
  }
  else if(name.find(".head.") != std::string::npos){
    new_name = name.substr(39);
    for(auto s = saved.begin(); s != saved.end(); ++s){
      if((s->first).find(new_name) != std::string::npos){
        value.copy_(s->second);
        return s->first;
      }
    }
    assert(false); 
  }
  else if(name.find("fc") != std::string::npos){
    new_name = "extractor_" + name.substr(name.find("fc"));
    value.copy_(saved.at(new_name));
    return new_name;
  }
  else{
    assert(false);
  }
}

std::string VoVNetMapper::rpn(const std::string& name, torch::Tensor& value, std::map<std::string, torch::Tensor>& saved){
  std::string new_name; 
  if(name.find("anchor") != std::string::npos){
    return name;
  }
  if(name.find("conv") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "rpn_conv.weight";
    else
      new_name = "rpn_conv.bias";
    value.copy_(saved.at(new_name));
    return new_name; 
  }
  else if(name.find("bbox") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "rpn_bbox.weight";
    else
      new_name = "rpn_bbox.bias";
    value.copy_(saved.at(new_name));
    // updated.insert(i.key());
    // mapping[i.key()] = name;
    return new_name;
  }
  else if(name.find("logits") != std::string::npos){
    if(name.find("weight") != std::string::npos)
      new_name = "rpn_logits.weight";
    else
      new_name = "rpn_logits.bias";
    value.copy_(saved.at(new_name));
    // updated.insert(name);
    // mapping[i.key()] = name;
    return new_name;
  }
  else{
    assert(false);
  } 
}

}
}