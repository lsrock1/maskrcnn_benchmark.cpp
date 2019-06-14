#include "segmentation_mask.h"
#include "mask.h"
#include "conv2d.h"
#include <algorithm>
#include <cassert>
#include <iostream>

namespace rcnn{
namespace structures{

torch::Tensor ArrayToTensor(char* mask, int h, int w, int n){
  torch::Tensor mask_tensor = torch::empty({h * w * n});
  float* data = mask_tensor.data<float>();
  for(size_t i = 0; i < h * w * n; ++i){
    data[i] = mask[i];
  }
  delete[] mask;
  return mask_tensor.reshape({h, w, n}).permute({2, 0, 1});
}

Polygons::Polygons(std::vector<std::vector<double>> polygons, std::pair<int, int> size, std::string mode)
                  :mode_(mode),
                   size_(size)
{
  for(auto& poly : polygons){
    torch::Tensor tmp = torch::empty({poly.size()}).to(torch::kF64);
    double* data = tmp.data<double>();
    for(size_t i = 0; i < poly.size(); ++i){
      data[i] = poly[i];
    }
    polygons_.push_back(tmp);
  }
  size_ = size;
  mode_ = mode;
}

Polygons::Polygons(std::vector<torch::Tensor> polygons, std::pair<int, int> size, std::string mode)
                  :mode_(mode),
                   size_(size)
{
  polygons_ = polygons;
  size_ = size;
  mode_ = mode;
}

Polygons Polygons::Transpose(Flip method){
  std::vector<torch::Tensor> flipped_polygons;
  int width = std::get<0>(size_), height = std::get<1>(size_);
  int dim, idx;
  if(method == FLIP_LEFT_RIGHT){
    dim = width;
    idx = 0;
  }
  else{
    dim = height;
    idx = 1;
  }
  for(auto& poly : polygons_){
    torch::Tensor p = poly.clone();
    p.slice(0, idx, -1, 2) = dim - poly.slice(0, idx, -1, 2) - 1;
    flipped_polygons.push_back(p);
  }
  return Polygons(flipped_polygons, size_, mode_);
}

Polygons Polygons::Crop(std::vector<int> box){
  int w = box[2] - box[0], h = box[3] - box[1];
  w = std::max(w, 1);
  h = std::max(h, 1);

  std::vector<torch::Tensor> cropped_polygons;
  for(auto& poly : polygons_){
    torch::Tensor p = poly.clone();
    p.slice(0, 0, -1, 2) = p.slice(0, 0, -1, 2) - box[0];
    p.slice(0, 1, -1, 2) = p.slice(0, 1, -1, 2) - box[1];
    cropped_polygons.push_back(p);
  }

  return Polygons(cropped_polygons, std::make_pair(w, h), mode_);
}

Polygons Polygons::Resize(std::pair<int, int> size){
  std::pair<float, float> ratios = std::make_pair(
    static_cast<float>(std::get<0>(size)) / static_cast<float>(std::get<0>(size_)),
    static_cast<float>(std::get<1>(size)) / static_cast<float>(std::get<1>(size_))
  );
  if(std::get<0>(ratios) == std::get<1>(ratios)){
    float ratio = std::get<0>(ratios);
    std::vector<torch::Tensor> scaled_polys;
    for(auto& poly : polygons_)
      poly.mul(ratio);
    return Polygons(scaled_polys, size, mode_);
  }
  float ratio_w = std::get<0>(ratios), ratio_h = std::get<1>(ratios);
  std::vector<torch::Tensor> scaled_polygons;
  for(auto& poly : polygons_){
    torch::Tensor p = poly.clone();
    p.slice(0, 0, -1, 2).mul_(ratio_w);
    p.slice(0, 1, -1, 2).mul_(ratio_h);
    scaled_polygons.push_back(p);
  }
  return Polygons(scaled_polygons, size, mode_);
}

torch::Tensor Polygons::GetMaskTensor(){
  int width = std::get<0>(size_), height = std::get<1>(size_);
  std::vector<std::vector<double>> pl;
  for(auto& poly : polygons_){
    std::vector<double> inner;
    for(int i = 0; i < poly.size(0); ++i)
      inner.push_back(poly.select(0, i).item<double>());  
    pl.push_back(inner);
  }
  std::vector<coco::RLEstr> rles = coco::frPoly(pl, height, width);
  coco::RLEstr rle = coco::merge(rles);
  torch::Tensor mask = ArrayToTensor(coco::decode(rle), std::get<0>(rle.size), std::get<1>(rle.size), rles.size());
  return mask;
}

std::ostream& operator << (std::ostream& os, const Polygons& bml){
  os << "Polygons(";
  os << "num_polygons=" << bml.polygons_.size();
  os << "image_width=" << std::get<0>(bml.size_);
  os << "image_height=" << std::get<1>(bml.size_);
  os << "mode=" << bml.mode_ << "\n";
  return os;
}

SegmentationMask::SegmentationMask(std::vector<std::vector<std::vector<double>>> polygons, std::pair<int, int> size, std::string mode){
  for(auto& poly : polygons){
    polygons_.emplace_back(poly, size, mode);
  }

  size_ = size;
  mode_ = mode;
}

SegmentationMask::SegmentationMask(std::vector<Polygons> polygons, std::pair<int, int> size, std::string mode){
  polygons_ = polygons;
  size_ = size;
  mode_ = mode;
}

SegmentationMask SegmentationMask::Transpose(Flip method){
  std::vector<Polygons> flipped;
  for(auto& poly : polygons_)
    flipped.push_back(poly.Transpose(method));

  return SegmentationMask(flipped, size_, mode_);
}

SegmentationMask SegmentationMask::Crop(std::vector<int> box){
  int w = box[2] - box[0], h = box[3] - box[1];
  std::vector<Polygons> cropped;
  for(auto& poly : polygons_)
    cropped.push_back(poly.Crop(box));
  return SegmentationMask(cropped, std::make_pair(w, h), mode_);
}

SegmentationMask SegmentationMask::Resize(std::pair<int, int> size){
  std::vector<Polygons> scaled;
  for(auto& poly : polygons_)
    scaled.push_back(poly.Resize(size));
  return SegmentationMask(scaled, size, mode_);
}

SegmentationMask SegmentationMask::to(){
  return *this;
}

SegmentationMask SegmentationMask::operator[](torch::Tensor item){
  assert(item.sizes().size() == 1);
  std::vector<Polygons> selected_polygons;
  if(item.dtype() == torch::kByte){
    int length = item.size(0);
    for(size_t i = 0; i < length; ++i){
      if(item.select(0, i).item<bool>())
        selected_polygons.push_back(polygons_[i]);
    }
    return SegmentationMask(selected_polygons, size_, mode_);
  }
  else{
    //index_select
    int length = item.size(0);
    for(size_t i = 0; i < length; ++i){
      selected_polygons.push_back(polygons_[item.select(0, i).item<int>()]);
    }
    return SegmentationMask(selected_polygons, size_, mode_);
  }
}

torch::Tensor SegmentationMask::GetMaskTensor(){
  std::vector<torch::Tensor> masks;
  for(auto& poly : polygons_)
    masks.push_back(poly.GetMaskTensor());
  return torch::stack(masks);
}

std::ostream& operator << (std::ostream& os, const SegmentationMask& bml){
  os << "SegmentationMask(";
  os << "num_instances=" << bml.polygons_.size() << ", ";
  os << "image_width=" << std::get<0>(bml.size_) << ", ";
  os << "image_height=" << std::get<1>(bml.size_) << "\n";
  return os;
}

// BinaryMaskList::BinaryMaskList(torch::Tensor masks, std::pair<int64_t, int64_t> size){
//   masks_ = masks.clone();
//   if(masks_.sizes().size() == 2)
//     masks_.unsqueeze(0);
//   size_ = size;
// }

// BinaryMaskList::BinaryMaskList(std::vector<torch::Tensor> masks, std::pair<int64_t, int64_t> size){
//   if(masks.size() == 0){
//     masks_ = torch::empty({0, std::get<1>(size), std::get<0>(size)});
//   }
//   else{
//     masks_ = torch::stack(masks, /*dim=*/2).clone();
//   }
//   if(masks_.sizes().size() == 2)
//     masks_.unsqueeze(0);
//   size_ = size;
// }

// BinaryMaskList::BinaryMaskList(std::vector<coco::RLEstr> masks, std::pair<int64_t, int64_t> size){
//   if(masks.size() == 0){
//     masks_ = torch::empty({0, std::get<1>(size), std::get<0>(size)});
//   }
//   else{
//     std::vector<std::pair<int64_t, int64_t>> rle_sizes;
//     for(auto& inst: masks)
//       rle_sizes.push_back(inst.size);
//     char* mask_array = coco::decode(masks);
//     torch::Tensor mask_tensor = ArrayToTensor(mask_array).permute({2, 0, 1});
//     //TODO all the sizes must be the same size:
//     int64_t rle_height = std::get<0>(rle_sizes[0]), rle_width = std::get<1>(rle_sizes[0]);

//     assert(masks.size(1) == rle_height);
//     assert(masks.size(2) == rle_width);

//     int64_t width = std::get<0>(size), height = std::get<1>(size);
//     if(width != rle_width || height != rle_height){
//       mask_tensor = rcnn::layers::interpolate(
//         mask_tensor.unsqueeze(0)_.to(torch::kF32),
//         torch::IntArrayRef{height, width} 
//       ).select(0, 0).to(mask_tensor.type());
//     }
//     masks_ = mask_tensor;
//   }
//   if(masks_.sizes().size() == 2)
//     masks_.unsqueeze(0);
//   size_ = size;
// }

// BinaryMaskList::BinaryMaskList(const BinaryMaskList& masks, std::pair<int64_t, int64_t> size){
//   masks_ = masks.masks.clone();
//   size_ = size;
//   if(masks_.sizes().size() == 2)
//     masks_.unsqueeze(0);
// }

// BinaryMaskList BinaryMaskList::Transpose(Flip method){
//   int dim = (method == FLIP_TOP_BOTTOM ? 1 : 2);
//   torch::Tensor flipped_mask = masks_.flip(dim);
//   return BinaryMaskList(flipped_mask, size_);
// }

// BinaryMaskList Crop(std::vector<torch::Tensor> box){

// }

// friend std::ostream& operator << (std::ostream& os, const BinaryMaskList& bml);

}
}