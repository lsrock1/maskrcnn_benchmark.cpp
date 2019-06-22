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
  Annotation();
  int64_t id;
  int image_id;
  int category_id;
  std::vector<std::vector<double>> segmentation;
  std::vector<int> counts;
  std::string compressed_rle;
  std::pair<int, int> size;
  float area;
  std::vector<float> bbox;
  bool iscrowd;
};

struct Image{
  Image(const Value& value);
  Image();
  // ~Image();
  // Image(const Image& other);
  // Image(Image&& other);
  // Image& operator=(const Image& other);
  // Image& operator=(Image&& other);
  int id;
  int width;
  int height;
  //char* file_name;
  std::string file_name;
};

struct Categories{
  Categories(const Value& value);
  Categories();
  int id;
  std::string name;
  std::string supercategory;
};

struct COCO{
  COCO(std::string annotation_file);
  COCO();
  void CreateIndex();
  std::vector<int64_t> GetAnnIds(const std::vector<int> imgIds = std::vector<int>{}, const std::vector<int> catIds = std::vector<int>{}, const std::vector<float> areaRng = std::vector<float>{}, Crowd iscrowd=none);
  //info
  std::vector<int> GetCatIds(const std::vector<std::string> catNms = std::vector<std::string>{}, const std::vector<std::string> supNms = std::vector<std::string>{}, const std::vector<int> catIds = std::vector<int>{});
  std::vector<Annotation> LoadAnns(std::vector<int64_t> ids);
  std::vector<Image> LoadImgs(std::vector<int> ids);
  COCO LoadRes(std::string res_file);
  COCO(const COCO& other);
  COCO(COCO&& other);
  COCO operator=(const COCO& other);
  COCO operator=(COCO&& other);

  Document dataset;
  std::map<int64_t, Annotation> anns;
  std::map<int, Image> imgs;
  std::map<int, Categories> cats;
  std::map<int, std::vector<Annotation>> imgToAnns;
  std::map<int, std::vector<int>> catToImgs;
};

}
