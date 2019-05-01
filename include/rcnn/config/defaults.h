#pragma once
#include <string>
#include <map>
#include <vector>
#include "yaml-cpp/yaml.h"

namespace rcnn{
namespace config{

YAML::Node GetDefaultCFG();
// YAML::Node SetConfigFromFile(std::string path);

}//configs
}//mrcn