#pragma once
#include <string>
#include <map>
#include <tuple>
#include <vector>
#include <torch/torch.h>
#include <cassert>
// #include <ATen/ATen.h>
#include <stdexcept>

namespace rcnn{
namespace structures{
  class BoxList{
    using XMin = torch::Tensor;
    using YMin = torch::Tensor;
    using XMax = torch::Tensor;
    using YMax = torch::Tensor;
    using Width = int64_t;
    using Height = int64_t;

    static const int kFlipLeftRight = 0;
    static const int kFlipTopBottom = 1;

  public:
    BoxList();
    BoxList(torch::Tensor bbox, std::pair<Width, Height> image_size, const char* mode="xyxy");
    BoxList(torch::Tensor bbox, std::pair<Width, Height> image_size, std::string mode="xyxy");
    void AddField(const std::string field_name, torch::Tensor field_data);
    torch::Tensor GetField(const std::string field_name);
    bool HasField(const std::string field_name);
    std::vector<std::string> Fields();
    BoxList Convert(const std::string mode);
    BoxList Resize(const std::pair<Width, Height> size);
    BoxList Transpose(const int flip_type);
    BoxList Crop(const std::tuple<int64_t, int64_t, int64_t, int64_t> box);
    BoxList To(const torch::Device device);
    int64_t Length() const;
    BoxList operator[](torch::Tensor item);
    BoxList operator[](const int64_t index);
        
    BoxList ClipToImage(const bool remove_empty=true);
    torch::Tensor Area();
    BoxList CopyWithFields(const std::vector<std::string> fields, const bool skip_missing=false);
    std::map<std::string, torch::Tensor> get_extra_fields() const;
    std::pair<Width, Height> get_size() const;
    torch::Device get_device() const;
    torch::Tensor get_bbox() const;
    std::string get_mode() const;
    void set_size(const std::pair<Width, Height> size);
    void set_extra_fields(const std::map<std::string, torch::Tensor> fields);
    void set_bbox(const torch::Tensor bbox);
    void set_mode(const std::string mode);
    void set_mode(const char* mode);

  private:
    std::map<std::string, torch::Tensor> extra_fields_;
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