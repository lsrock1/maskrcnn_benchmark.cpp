#include "segmentation_mask.h"


namespace rcnn{
namespace structures{

BinaryMaskList::BinaryMaskList(torch::Tensor masks, std::pair<int64_t, int64_t> size){
  masks_ = masks.clone();
}

BinaryMaskList::BinaryMaskList(std::vector<torch::Tensor> masks, std::pair<int64_t, int64_t> size){
  if(masks.size() == 0){
    masks_ = torch::empty({0, std::get<1>(size), std::get<0>(size)});
  }
  else{
    masks_ = torch::stack(masks, /*dim=*/2).clone();
  }
}

class BinaryMaskList{
  public:
    BinaryMaskList(torch::Tensor masks, std::pair<int64_t, int64_t> size);
    BinaryMaskList(std::vector<torch::Tensor> masks, std::pair<int64_t, int64_t> size);
    BinaryMaskList(const BinaryMaskList& masks, std::pair<int64_t, int64_t> size);
    BinaryMaskList(const BinaryMaskList& masks) = default;
    ~BinaryMaskList() = default;
    //BinaryMaskList RLE version //not implemented now
    BinaryMaskList Transpose(int method);
    BinaryMaskList Crop(std::vector<torch::Tensor>);
    BinaryMaskList Crop(torch::Tensor);
    BinaryMaskList Resize(std::pair<int64_t, int64_t> size);
    BinaryMaskList Resize(float size);
    BinaryMaskList Resize(int64_t size);
    PolygonList ConvertToPolygon();
    BinaryMaskList to();
    int Length();
    BinaryMaskList operator[](torch::Tensor index);

  private:
    FindContours();//TODO

};

friend std::ostream& operator << (std::ostream& os, const BinaryMaskList& bml);

class PolygonInstance{
  public:
    PolygonInstance(std::vector<std::vector<int64_t>> polygons, std::pair<int64_t, int64_t> size);
    PolygonInstance(PolygonInstance polygons, std::pair<int64_t, int64_t> size);

  private:
    
}

}
}