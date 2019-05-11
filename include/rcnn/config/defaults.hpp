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

template<typename T>
void SetNode(YAML::Node parent, T value){
  if(parent.Type() == YAML::NodeType::Undefined){
    parent = value;
  }
}
//TODO
//std::string argument occurs undefined reference error
//so wrapper and return char*

//get code related bug
//https://github.com/pytorch/pytorch/issues/19353
//implementation in header occurs link error
template<typename T>
const T GetCFG(std::initializer_list<const char*> node);
class CFGString{
  private:
    std::string name_;

  public:
    CFGString(std::string name);
    const char* get();
};

}//configs
}//mrcn