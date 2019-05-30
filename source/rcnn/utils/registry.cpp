#include "registry.h"
#include <map>
#include "backbone/backbone.h"
#include <cassert>

namespace rcnn{
namespace registry{

backbone BACKBONES(std::string conv_body){
  std::map<std::string, backbone> backbone_builder_map{
    {"R-50-C4", rcnn::modeling::BuildResnetBackbone},
    {"R-50-C5", rcnn::modeling::BuildResnetBackbone},
    {"R-101-C4", rcnn::modeling::BuildResnetBackbone},
    {"R-101-C5", rcnn::modeling::BuildResnetBackbone},
    {"R-50-FPN", rcnn::modeling::BuildResnetFPNBackbone},
    {"R-101-FPN", rcnn::modeling::BuildResnetFPNBackbone},
    {"R-152-FPN", rcnn::modeling::BuildResnetFPNBackbone}
  };
  assert(backbone_builder_map.count(conv_body));
  return backbone_builder_map.find(conv_body)->second;
}

}
}

