#pragma once
#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"

namespace rcnn{
namespace config{
  namespace{
    YAML::Node* cfg = nullptr;
  };
void SetCFGFromFile(const char* file_path);

const YAML::Node* GetDefaultCFG();

template<typename T>
void SetNode(YAML::Node parent, T value){
  if(parent.Type() == YAML::NodeType::Undefined){
    parent = value;
  }
}

template<typename T>
const T GetCFG(std::initializer_list<const char*> node);
}//configs
}//mrcn