#pragma once
#include <string>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include <cassert>

namespace rcnn {
namespace config {

namespace {
 YAML::Node* cfg;
}

template<typename T>
void SetNode(YAML::Node parent, T value) {
  if (parent.Type() == YAML::NodeType::Undefined) {
    parent = value;
  }
};

void SetCFGFromFile(const std::string& file_path);

//cannot access cfg if implement here
template<typename T>
T GetCFG(std::initializer_list<const char*> node);

} // namespace config
} // namespace rcnn
