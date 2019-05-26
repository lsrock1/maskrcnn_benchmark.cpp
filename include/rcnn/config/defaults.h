#pragma once
#include <string>
#include <algorithm>
#include "yaml-cpp/yaml.h"

namespace rcnn{
namespace config{

namespace{
  YAML::Node* cfg = nullptr;
};

void SetCFGFromFile(const char* file_path);

template<typename T>
void SetNode(YAML::Node parent, T value){
  if(parent.Type() == YAML::NodeType::Undefined){
    parent = value;
  }
}

class CFGS{
  private:
    std::string name_;

  public:
    CFGS(std::string name);
    const char* get();
};

//TODO
//std::string argument occurs undefined reference error
//so seperate get cfg char

//get code related bug
//https://github.com/pytorch/pytorch/issues/19353
//implementation in header occurs link error
template<typename T>
T GetCFG(std::initializer_list<const char*> node);

}//configs
}//mrcn