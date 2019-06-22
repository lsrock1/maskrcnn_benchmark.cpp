#pragma once
#include "mask_api.h"
#include <string>
#include <vector>


namespace coco{

struct RLEstr{
  RLEstr(std::pair<coco::siz, coco::siz> size_, std::string counts_) : size(size_), counts(counts_){};
  RLEstr() :size(std::make_pair(0, 0)), counts(""){};
  std::pair<coco::siz, coco::siz> size;
  std::string counts;
};
//coco namespace means "from mask_api.h"
struct RLEs{
  RLEs(coco::siz n = 0);
  RLEs(const RLEs& other);
  RLEs(RLEs&& other);
  RLEs& operator=(const RLEs& other);
  RLEs& operator=(RLEs&& other);
  ~RLEs();
  RLEs();
  coco::siz operator[](std::string key);
  std::vector<RLEstr> toString();
  coco::RLE* _R;
  coco::siz _n;
};

struct Masks{
  Masks(coco::siz h, coco::siz w, coco::siz n);
  coco::byte* _mask;
  coco::siz _n, _w, _h;
};

RLEs _frString(std::vector<RLEs>& rleObjs);
RLEstr merge(std::vector<RLEstr>& rleObjs, int intersect = 0);
std::vector<RLEstr> frPoly(std::vector<std::vector<double>>& polygon, int h, int w);
std::vector<RLEstr> encode(byte* mask, int h, int w, int n);
coco::Masks decode(RLEstr rleObjs);
std::vector<int64_t> area(std::vector<RLEstr>& rleObjs);
std::vector<double> toBbox(std::vector<RLEstr>& rleObjs);
}