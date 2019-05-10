#pragma once
#include <string>
#include <map>
#include <vector>
#include "yaml-cpp/yaml.h"

namespace rcnn{
namespace config{
  namespace{
    YAML::Node* cfg = nullptr;
  }
void SetDefaultCFGFromFile(std::string file_path);
const YAML::Node* GetDefaultCFG();
// YAML::Node SetConfigFromFile(std::string path);

}//configs
}//mrcn