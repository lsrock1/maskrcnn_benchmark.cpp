#pragma once
#include "rapidjson/document.h"
#include <map>
#include <string>
#include <vector>


namespace coco{

using namespace rapidjson;

enum Crowd{
  none = -1,
  F = 0,
  T = 1
};

struct Annotation{
  Annotation(const Value& value);
  Annotation() = default;
  int id;
  int image_id;
  int category_id;
  std::vector<std::vector<float>> segmentation;
  float area;
  std::vector<float> bbox;
  bool iscrowd;
};

struct Image{
  Image(const Value& value);
  Image() = default;
  int id;
  int width;
  int height;
  std::string file_name;
};

struct Categories{
  Categories(const Value& value);
  Categories() = default;
  int id;
  std::string name;
  std::string supercategory;
};

class COCO{
  public:
    COCO(const std::string annotation_file);
    COCO();
    void CreateIndex();
    std::vector<int> GetAnnIds(const std::vector<int> imgIds = std::vector<int>{}, const std::vector<int> catIds = std::vector<int>{}, const std::vector<float> areaRng = std::vector<float>{}, Crowd iscrowd=none);
    //info
    std::vector<int> GetCatIds(const std::vector<std::string> catNms = std::vector<std::string>{}, const std::vector<std::string> supNms = std::vector<std::string>{}, const std::vector<int> catIds = std::vector<int>{});
    std::vector<Annotation> LoadAnns(std::vector<int> ids);
    std::vector<Image> LoadImgs(std::vector<int> ids);

  private:
    Document dataset;
    std::map<int, Annotation> anns;
    std::map<int, Image> imgs;
    std::map<int, Categories> cats;
    std::map<int, std::vector<Annotation>> imgToAnns;
    std::map<int, std::vector<int>> catToImgs;
};

}
