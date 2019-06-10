#pragma once
#include "mask_api.h"


namespace coco{
//coco namespace means "from mask_api.h"
struct RLEstr{
  std::pair<coco::siz, coco::siz> size;
  std::string counts;
};

class RLEs{
  public:
    RLEs();
    ~RLEs();
    coco::siz operator[](std::string key);
    RLEstr toString();

  private:
    coco::RLE* _R;
    coco::size _n
};

}