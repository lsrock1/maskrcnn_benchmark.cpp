#pragma once
#include <torch/torch.h>
#include "mask.h"
// #include <opencv4/opencv.hpp>


namespace rcnn{
namespace structures{

enum Flip{
  FLIP_LEFT_RIGHT,
  FLIP_TOP_BOTTOM
};

torch::Tensor ArrayToTensor(coco::Masks mask);

class Polygons{

public:
  Polygons(std::vector<std::vector<double>> polygons, std::pair<int, int> size, std::string mode);
  Polygons(std::vector<torch::Tensor> polygons, std::pair<int, int> size, std::string mode);
  Polygons Transpose(Flip method);
  Polygons Crop(std::vector<int> box);
  Polygons Resize(std::pair<int, int> size);
  // Polygons Convert(std::string mode);
  torch::Tensor GetMaskTensor();

private:
  std::pair<int, int> size_;
  std::string mode_;
  std::vector<torch::Tensor> polygons_;

friend std::ostream& operator << (std::ostream& os, const Polygons& bml);

};

class SegmentationMask{

public:
  SegmentationMask(std::vector<std::vector<std::vector<double>>> polygons, std::pair<int, int> size, std::string mode);
  SegmentationMask(std::vector<Polygons> polygons, std::pair<int, int> size, std::string mode);
  SegmentationMask Transpose(Flip method);
  SegmentationMask Crop(std::vector<int> box);
  SegmentationMask Resize(std::pair<int, int> size);
  SegmentationMask to();
  torch::Tensor GetMaskTensor();

  SegmentationMask operator[](torch::Tensor item);

private:
  std::vector<Polygons> polygons_;
  std::pair<int, int> size_;
  std::string mode_;

friend std::ostream& operator << (std::ostream& os, const SegmentationMask& bml);

};


// class BinaryMaskList{
//   public:
//     BinaryMaskList(torch::Tensor masks, std::pair<int64_t, int64_t> size);
//     BinaryMaskList(std::vector<torch::Tensor> masks, std::pair<int64_t, int64_t> size);
//     BinaryMaskList(std::vector<coco::Annotation> masks, std::pair<int64_t, int64_t> size);
//     BinaryMaskList(const BinaryMaskList& masks, std::pair<int64_t, int64_t> size);
//     BinaryMaskList(const BinaryMaskList& masks) = default;
//     ~BinaryMaskList() = default;
//     //BinaryMaskList RLE version //not implemented now
//     BinaryMaskList Transpose(Flip method);
//     BinaryMaskList Crop(std::vector<int> box);
//     BinaryMaskList Crop(torch::Tensor box);
//     BinaryMaskList Resize(std::pair<int64_t, int64_t> size);
//     BinaryMaskList Resize(float size);
//     BinaryMaskList Resize(int64_t size);
//     PolygonList ConvertToPolygon();
//     BinaryMaskList to();
//     int Length();
//     BinaryMaskList operator[](torch::Tensor index);
//     torch::Tensor masks_;
//     std::pair<int64_t, int64_t> size_;

//   private:
//     FindContours();//TODO

// };

// friend std::ostream& operator << (std::ostream& os, const BinaryMaskList& bml);

// class PolygonInstance{
//   public:
//     PolygonInstance(std::vector<std::vector<int64_t>> polygons, std::pair<int64_t, int64_t> size);
//     PolygonInstance(PolygonInstance polygons, std::pair<int64_t, int64_t> size);

//   private:
    
// };

}
}