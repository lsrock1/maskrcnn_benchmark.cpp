#pragma once
#include <string>
#include <map>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include <stdexcept>
#include "segmentation_mask.h"
#include "mask.h"

namespace rcnn {
namespace structures {

namespace {
  using XMin = torch::Tensor;
  using YMin = torch::Tensor;
  using XMax = torch::Tensor;
  using YMax = torch::Tensor;
  using Width = int64_t;
  using Height = int64_t;
}

class BoxList {

public:
  BoxList();
  ~BoxList();
  BoxList(const BoxList& other);
  BoxList& operator=(const BoxList& other);
  BoxList(BoxList&& other) noexcept;
  BoxList& operator=(BoxList&& other) noexcept;

  BoxList(torch::Tensor bbox, std::pair<Width, Height> image_size, std::string mode="xyxy");
  
  void AddField(const std::string field_name, torch::Tensor field_data);
  void AddField(const std::string field_name, std::vector<coco::RLEstr> rles);
  void AddField(const std::string field_name, rcnn::structures::SegmentationMask* masks);

  template<typename T = torch::Tensor>
  T GetField(const std::string field_name){
    return extra_fields_.find(field_name)->second;
  };
  rcnn::structures::SegmentationMask* GetMasksField(const std::string field_name);
  bool HasField(const std::string field_name);
  std::vector<std::string> Fields();
  BoxList Convert(const std::string mode);
  BoxList Resize(const std::pair<Width, Height> size);
  BoxList Transpose(const Flip flip_type);
  BoxList Crop(const std::tuple<int64_t, int64_t, int64_t, int64_t> box);
  BoxList To(const torch::Device device);
  int64_t Length() const;
  BoxList nms(const float nms_thresh, const int max_proposals=-1, const std::string score_field="scores");
  BoxList RemoveSmallBoxes(const int min_size);
  BoxList operator[](torch::Tensor item);
  BoxList operator[](const int64_t index);
      
  BoxList ClipToImage(const bool remove_empty=true);
  torch::Tensor Area();
  BoxList CopyWithFields(const std::vector<std::string> fields, const bool skip_missing=false);
  std::map<std::string, torch::Tensor> get_extra_fields() const;
  std::vector<coco::RLEstr> get_rles() const;
  std::pair<Width, Height> get_size() const;
  SegmentationMask* get_masks() const;
  torch::Device get_device() const;
  torch::Tensor get_bbox() const;
  std::string get_mode() const;
  void set_rles(const std::vector<coco::RLEstr> rles);
  void set_size(const std::pair<Width, Height> size);
  void set_masks(SegmentationMask* masks);
  void set_extra_fields(const std::map<std::string, torch::Tensor> fields);
  void set_bbox(const torch::Tensor bbox);
  void set_mode(const std::string mode);
  void set_mode(const char* mode);
  
  static torch::Tensor BoxListIOU(BoxList a, BoxList b);
  static BoxList CatBoxList(std::vector<BoxList> boxlists);

private:
  std::map<std::string, torch::Tensor> extra_fields_;
  std::vector<coco::RLEstr> rles_;
  rcnn::structures::SegmentationMask* masks_{nullptr};
  torch::Device device_;
  torch::Tensor bbox_;
  std::pair<int64_t, int64_t> size_;
  std::string mode_;
  void CopyExtraFields(const BoxList box);
  std::tuple<XMin, YMin, XMax, YMax> SplitIntoXYXY();

friend std::ostream& operator << (std::ostream& os, const BoxList& bbox);
};

}
}