#include "mask_api.h"
#include "mask.h"
#include "coco.h"
#include <cassert>
#include <iostream>


namespace coco{
//coco namespace means "from mask_api.h"
RLEs::RLEs(): _n(0), _R(nullptr){
}

RLEs::RLEs(coco::siz n){
  _n = n;
  coco::rlesInit(&_R, n);
}

RLEs::RLEs(const RLEs& other){
  _n = other._n;
  coco::rlesInit(&_R, _n);
  for(size_t i = 0; i < _n; ++i){
    coco::rleInit(&_R[i], other._R[i].h, other._R[i].w, other._R[i].m, other._R[i].cnts);
  }
}

RLEs::RLEs(RLEs&& other){
  _n = other._n;

  if(_R){
    for(size_t i = 0; i < _n; ++i){
      if(_R[i].cnts)delete[] _R[i].cnts;
    }
    delete[] _R;
  }

  _R = other._R;
  other._R = nullptr;
}

RLEs& RLEs::operator=(const RLEs& other){
  if(this != &other){
    _n = other._n;
    coco::rlesInit(&_R, _n);
    for(size_t i = 0; i < _n; ++i){
      coco::rleInit(&_R[i], other._R[i].h, other._R[i].w, other._R[i].m, other._R[i].cnts);
    }
  }
  return *this;
}

RLEs& RLEs::operator=(RLEs&& other){
  if(this != &other){
    _n = other._n;
    if(_R){
      for(size_t i = 0; i < _n; ++i)
        delete[] _R[i].cnts;
      delete[] _R;
    }

    _R = other._R;
    other._R = nullptr;
  }
  return *this;
}

RLEs::~RLEs(){
  if(_R){
    for(size_t i = 0; i < _n; ++i)
      delete[] _R[i].cnts;
    delete[] _R;
  }
}

coco::siz RLEs::operator[](std::string key){
  if(key.compare("n") == 0){
    return _n;
  }
  else
    assert(false);
}

std::vector<RLEstr> RLEs::toString(){
  std::vector<RLEstr> objs;
  for(size_t i = 0; i < _n; ++i){
    char* c_string = coco::rleToString(&_R[i]);
    RLEstr obj = RLEstr();
    obj.size = std::make_pair(_R[i].h, _R[i].w);
    obj.counts = std::string (c_string);
    delete[] c_string;
    objs.push_back(obj);
  }
  
  return objs;
}

Masks::Masks(coco::siz h, coco::siz w, coco::siz n){
  _mask = new coco::byte[h*w*n];
  _n = n;
  _h = h;
  _w = w;
}

RLEs _frString(std::vector<RLEstr>& rleObjs){
  size_t n = rleObjs.size();
  RLEs Rs = RLEs(n);
  for(size_t i = 0; i < n; ++i)
    coco::rleFrString(&Rs._R[i], rleObjs[i].counts, std::get<0>(rleObjs[i].size), std::get<1>(rleObjs[i].size));

  return Rs;
}

RLEstr merge(std::vector<RLEstr>& rleObjs, int intersect){
  RLEs Rs = _frString(rleObjs);
  RLEs R = RLEs(1);
  coco::rleMerge(Rs._R, R._R, Rs._n, intersect);
  return R.toString()[0];
}

std::vector<RLEstr> frPoly(std::vector<std::vector<double>>& polygon, int h, int w){
  int n = polygon.size();
  RLEs Rs = RLEs(n);
  for(size_t i = 0; i < n; ++i){
    double* tmp = new double[polygon[i].size()];
    for(int j = 0; j < polygon[i].size(); ++j){
      tmp[j] = polygon[i][j];
    }
    coco::rleFrPoly(&(Rs._R[i]), tmp, static_cast<int>(polygon[i].size()/2), h, w);
    delete[] tmp;
  }
  return Rs.toString();
}

Masks decode(RLEstr rleObjs){
  std::vector<RLEstr> rle_vec{rleObjs};
  RLEs Rs = _frString(rle_vec);
  siz h = Rs._R[0].h, w = Rs._R[0].w, n = Rs._n;
  Masks masks = Masks(h, w, n);
  coco::rleDecode(Rs._R, masks._mask, n);
  return masks;
}

std::vector<RLEstr> encode(byte* mask, int h, int w, int n){
  RLEs Rs = RLEs(n);
  coco::rleEncode(Rs._R, mask, h, w, n);
  return Rs.toString();
}

std::vector<int64_t> area(std::vector<RLEstr>& rleObjs){
  RLEs Rs = _frString(rleObjs);
  uint* _a = new uint[Rs._n];
  rleArea(Rs._R, Rs._n, _a);
  std::vector<int64_t> area;
  for(size_t i = 0; i < Rs._n; ++i)
    area.push_back(static_cast<int64_t>(_a[i]));
  delete[] _a;
  return area;
}

std::vector<double> toBbox(std::vector<RLEstr>& rleObjs){
  RLEs Rs = _frString(rleObjs);
  siz n = Rs._n;
  BB _bb = new double[4*n]; //<BB> malloc(4*n* sizeof(double))
  rleToBbox(Rs._R, _bb, n);
  
  std::vector<double> bbox;
  for(size_t i = 0; i < 4*n; ++i)
    bbox.push_back(_bb[i]);
  delete[] _bb;
  return bbox;
}

}