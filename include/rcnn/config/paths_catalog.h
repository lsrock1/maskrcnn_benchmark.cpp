#pragma once
#include <map>
#include <string>


namespace rcnn{
namespace config{

using args = std::map<std::string, std::string>;

class DatasetCatalog{

public:
  DatasetCatalog(){};
  std::tuple<std::string, std::string, std::string> operator[](std::string name);

  static const std::string DATA_DIR;
  static const std::map<std::string, args> DATASETS;
  //no keypoints and voc

};

class ModelCatalog{

public:
  ModelCatalog(){};
  static const std::string S3_C2_DETECTRON_URL;
  static const std::map<std::string, std::string> C2_IMAGENET_MODELS;
  static const std::string C2_DETECTRON_SUFFIX;
  static const std::map<std::string, std::string> C2_DETECTRON_MODELS;
  
  static std::string get(std::string name);
  static std::string get_c2_imagenet_pretrained(std::string name);
  static std::string get_c2_detectron_12_2017_baselines(std::string name);
};

}
}