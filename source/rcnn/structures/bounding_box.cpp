#include "bounding_box.h"
#include <nms.h>
#include <cat.h>
#include <box_iou.h>

#include <cassert>
#include <algorithm>
#include <iostream>


namespace rcnn{
namespace structures{

torch::Tensor BoxList::BoxListIOU(BoxList a, BoxList b){
  assert(a.get_size() == b.get_size());
  return rcnn::layers::box_iou(a.Area(), b.Area(), a.get_bbox(), b.get_bbox());
  // int TO_REMOVE = 1;
  // a = a.Convert("xyxy");
  // b = b.Convert("xyxy");
  // torch::Tensor area_a = a.Area();
  // torch::Tensor area_b = b.Area();
  // torch::Tensor bbox_a = a.get_bbox();
  // torch::Tensor bbox_b = b.get_bbox();
  // torch::Tensor lt = torch::max(bbox_a.unsqueeze(1).slice(/*dim=*/2, /*start=*/0, /*end=*/2), bbox_b.slice(1, 0, 2));
  // torch::Tensor rb = torch::min(bbox_a.unsqueeze(1).slice(/*dim=*/2, /*start=*/2, /*end=*/4), bbox_b.slice(1, 2, 4));
  // torch::Tensor wh = (rb - lt + TO_REMOVE).clamp(0);
  // torch::Tensor inter = wh.select(2, 0) * wh.select(2, 1);
  // return inter / (area_a.unsqueeze(1) + area_b - inter);
}

BoxList BoxList::CatBoxList(std::vector<BoxList> boxlists){
  std::pair<Width, Height> size = boxlists[0].get_size();
  std::string mode = boxlists[0].get_mode();
  std::vector<std::string> fields = boxlists[0].Fields();
  std::vector<std::string> compared_field;
  std::sort(fields.begin(), fields.end());
  std::vector<torch::Tensor> cat_bbox;
  
  for(auto& boxlist: boxlists){
    compared_field = boxlist.Fields();
    std::sort(compared_field.begin(), compared_field.end());
    cat_bbox.push_back(boxlist.get_bbox());
    assert(boxlist.get_size() == size);
    assert(boxlist.get_mode() == mode);
    assert(fields == compared_field);
  }
  BoxList cat_boxlists = BoxList(rcnn::layers::cat(cat_bbox, 0), size, mode);
  for(auto& field: fields){
    std::vector<torch::Tensor> cat_field;
    for(auto& boxlist: boxlists){
      cat_field.push_back(boxlist.GetField(field));
    }
    cat_boxlists.AddField(field, rcnn::layers::cat(cat_field, 0));
    cat_field.clear();
  }
  return cat_boxlists;
}

BoxList::BoxList(): size_(std::make_pair(0, 0)), mode_("xyxy"), device_("cpu"){}

//size<width, height>
BoxList::BoxList(torch::Tensor bbox, std::pair<Width, Height> image_size, std::string mode)
    : device_(bbox.device()),
      size_(image_size),
      bbox_(bbox),
      mode_(mode){};

BoxList::~BoxList(){
  if(masks_)
    delete masks_;
}

BoxList::BoxList(const BoxList& other) :device_(other.device_){
  size_ = other.size_;
  bbox_ = other.bbox_;
  mode_ = other.mode_;
  extra_fields_ = other.extra_fields_;
  rles_ = other.rles_;
  if(other.masks_)
    masks_ = new SegmentationMask(*other.masks_);
}

BoxList& BoxList::operator=(const BoxList& other){
  if(this != &other){
    if(masks_)
      delete masks_;
    device_ = other.device_;
    size_ = other.size_;
    bbox_ = other.bbox_;
    mode_ = other.mode_;
    extra_fields_ = other.extra_fields_;
    rles_ = other.rles_;
    if(other.masks_)
      masks_ = new SegmentationMask(*other.masks_);
  }
  return *this;
}

BoxList::BoxList(BoxList&& other) :device_(other.device_){
  device_ = other.device_;
  size_ = other.size_;
  bbox_ = other.bbox_;
  mode_ = other.mode_;
  extra_fields_ = other.extra_fields_;
  rles_ = other.rles_;
  if(other.masks_){
    masks_ = other.masks_;
    other.masks_ = nullptr;
  }
}

BoxList& BoxList::operator=(BoxList&& other){
  if(this != &other){
    if(masks_)
      delete masks_;
    device_ = other.device_;
    size_ = other.size_;
    bbox_ = other.bbox_;
    mode_ = other.mode_;
    extra_fields_ = other.extra_fields_;
    rles_ = other.rles_;
    if(other.masks_){
      masks_ = other.masks_;
      other.masks_ = nullptr;
    }
  }
  return *this;
}

//only supports tensor field data
void BoxList::AddField(const std::string field_name, torch::Tensor field_data){
  extra_fields_[field_name] = field_data;
  if(field_name.compare("mask") == 0)
    rles_.clear();
}

void BoxList::AddField(const std::string field_name, std::vector<coco::RLEstr> rles){
  rles_ = rles;
  if(extra_fields_.count(field_name))
    extra_fields_.erase(field_name);
}

void BoxList::AddField(const std::string field_name, rcnn::structures::SegmentationMask* masks){
  assert(field_name.compare("masks") == 0);
  if(masks_)
    delete masks_;
  masks_ = masks;
}


template<>
std::vector<coco::RLEstr> BoxList::GetField(const std::string field_name){
  return rles_;
}

rcnn::structures::SegmentationMask* BoxList::GetMasksField(const std::string field_name){
  return masks_;
}

bool BoxList::HasField(const std::string field_name){
  if(field_name.compare("mask") == 0 && rles_.size() > 0)
    return true;
  
  if(field_name.compare("masks") == 0 && masks_)
    return true;

  return extra_fields_.count(field_name) ? true : false;
}

std::vector<std::string> BoxList::Fields(){
    std::vector<std::string> keys;
    for(auto i = extra_fields_.begin(); i != extra_fields_.end(); i++)
      keys.push_back(i->first);
    if(extra_fields_.count("mask") == 0 && rles_.size() > 0)
      keys.push_back("mask");
    if(masks_)
      keys.push_back("masks");
    return keys;
}

BoxList BoxList::Convert(const std::string mode){
  assert(mode.compare("xyxy") == 0 || mode.compare("xywh") == 0);
  if(mode_.compare(mode) != 0){
    std::tuple<XMin, YMin, XMax, YMax> splitted_box_coordinates = SplitIntoXYXY();
    torch::Tensor bbox_tensor;
    if(mode.compare("xyxy") == 0){
        bbox_tensor = torch::cat({
            std::get<0>(splitted_box_coordinates),
            std::get<1>(splitted_box_coordinates),
            std::get<2>(splitted_box_coordinates),
            std::get<3>(splitted_box_coordinates),
        }, -1);
    }
    else{
        int TO_REMOVE = 1;
        bbox_tensor = torch::cat({
            std::get<0>(splitted_box_coordinates),
            std::get<1>(splitted_box_coordinates),
            std::get<2>(splitted_box_coordinates) - std::get<0>(splitted_box_coordinates) + TO_REMOVE,
            std::get<3>(splitted_box_coordinates) - std::get<1>(splitted_box_coordinates) + TO_REMOVE
        }, -1);
    }
    BoxList bbox = BoxList(bbox_tensor, size_, mode);
    bbox.CopyExtraFields(*this);
    return bbox;
  }
  else{
    return *this;
  }
}

std::tuple<XMin, YMin, XMax, YMax> BoxList::SplitIntoXYXY(){
  if(this->mode_.compare("xyxy") == 0){
    std::vector<torch::Tensor> splitted_box_coordinates = bbox_.split(1, /*dim=*/-1);
    return std::make_tuple(
      splitted_box_coordinates.at(0),
      splitted_box_coordinates.at(1),
      splitted_box_coordinates.at(2),
      splitted_box_coordinates.at(3));
  }
  else if(mode_.compare("xywh") == 0){
    int TO_REMOVE = 1;
    std::vector<torch::Tensor> splitted_box_coordinates = bbox_.split(1, /*dim=*/-1);
    return std::make_tuple(
      splitted_box_coordinates.at(0),
      splitted_box_coordinates.at(1),
      splitted_box_coordinates.at(0) + (splitted_box_coordinates.at(2) - TO_REMOVE).clamp_min(0),
      splitted_box_coordinates.at(1) + (splitted_box_coordinates.at(3) - TO_REMOVE).clamp_min(0)
    );
  }
  else{
    assert(false);
  }
}

void BoxList::CopyExtraFields(const BoxList bbox){
  if(!bbox.get_extra_fields().empty()){
    auto extra_field_src = bbox.get_extra_fields();
    for(auto i = extra_field_src.begin(); i != extra_field_src.end(); ++i){
      extra_fields_[i->first] = i->second;
    }
  }
  if(bbox.get_rles().size() > 0)
    rles_ = bbox.get_rles();

  // if(bbox.get_masks())
  //   masks_ = new SegmentationMask(*bbox.get_masks());
}

BoxList BoxList::Resize(const std::pair<Width, Height> size){
  //width, height
  std::pair<float, float> ratios = std::make_pair(float(size.first) / float(size_.first), float(size.second) / float(size_.second));
  torch::Tensor scaled_bbox;
  if(ratios.first == ratios.second){
    float ratio = ratios.first;
    scaled_bbox = bbox_ * ratio;
  }
  else{
    float ratio_width = ratios.first, ratio_height = ratios.second;
    std::tuple<XMin, YMin, XMax, YMax> splitted_box_coordinates = SplitIntoXYXY();
    XMin scaled_xmin = std::get<0>(splitted_box_coordinates) * ratio_width;
    XMax scaled_xmax = std::get<2>(splitted_box_coordinates) * ratio_width;
    YMin scaled_ymin = std::get<1>(splitted_box_coordinates) * ratio_height;
    YMax scaled_ymax = std::get<3>(splitted_box_coordinates) * ratio_height;
    scaled_bbox = torch::cat({
        scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax
    }, -1);
  }
  BoxList bbox = BoxList(scaled_bbox, size, "xyxy");
  // bbox.CopyExtraFields(*this);
  for(auto i = extra_fields_.begin(); i != extra_fields_.end(); i++)
    bbox.AddField(i->first, i->second);
  if(masks_)
    bbox.set_masks(new SegmentationMask(masks_->Resize(size)));
  return bbox.Convert(get_mode());
}

BoxList BoxList::Transpose(const Flip method){
  int image_width = size_.first, image_height = size_.second;
  std::tuple<XMin, YMin, XMax, YMax> splitted_box_coordinates = SplitIntoXYXY();
  XMin transposed_xmin;
  XMax transposed_xmax;
  YMin transposed_ymin;
  YMax transposed_ymax;
  if(method == FLIP_LEFT_RIGHT){
    int TO_REMOVE = 1;
    transposed_xmin = image_width - std::get<2>(splitted_box_coordinates) - TO_REMOVE;
    transposed_xmax = image_width - std::get<0>(splitted_box_coordinates) - TO_REMOVE;
    transposed_ymin = std::get<1>(splitted_box_coordinates);
    transposed_ymax = std::get<3>(splitted_box_coordinates);
  }
  else{
    transposed_xmin = std::get<0>(splitted_box_coordinates);
    transposed_xmax = std::get<2>(splitted_box_coordinates);
    transposed_ymin = image_height - std::get<3>(splitted_box_coordinates);
    transposed_ymax = image_height - std::get<1>(splitted_box_coordinates);
  }

  torch::Tensor transposed_boxes = torch::cat({
    transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax
  }, -1);
  BoxList bbox = BoxList(transposed_boxes, size_, "xyxy");
  for(auto i = extra_fields_.begin(); i != extra_fields_.end(); i++)
    bbox.AddField(i->first, i->second);
  if(masks_)
    bbox.set_masks(new SegmentationMask(masks_->Transpose(method)));
  
  return bbox.Convert(mode_);
}

BoxList BoxList::Crop(const std::tuple<int64_t, int64_t, int64_t, int64_t> box){
  std::tuple<XMin, YMin, XMax, YMax> splitted_box_coordinates = SplitIntoXYXY();
  int64_t width = std::get<2>(box) - std::get<0>(box), height = std::get<3>(box) - std::get<1>(box);
  XMin cropped_xmin = (std::get<0>(splitted_box_coordinates) - std::get<0>(box)).clamp(/*min*/0, /*max*/width);
  YMin cropped_ymin = (std::get<1>(splitted_box_coordinates) - std::get<1>(box)).clamp(/*min*/0, /*max*/height);
  XMax cropped_xmax = (std::get<2>(splitted_box_coordinates) - std::get<0>(box)).clamp(/*min*/0, /*max*/width);
  YMax cropped_ymax = (std::get<3>(splitted_box_coordinates) - std::get<1>(box)).clamp(/*min*/0, /*max*/height);

  torch::Tensor cropped_box = torch::cat(
      {cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax}, -1
  );
  BoxList bbox = BoxList(cropped_box, std::make_pair(width, height), "xyxy");
  for(auto i = extra_fields_.begin(); i != extra_fields_.end(); i++)
    bbox.AddField(i->first, i->second);
  if(masks_)
    bbox.set_masks(new SegmentationMask(masks_->Crop(box)));
  return bbox.Convert(mode_);
}

BoxList BoxList::To(const torch::Device device){
  BoxList bbox = BoxList(bbox_.to(device), size_, mode_);
  for(auto i = extra_fields_.begin(); i != extra_fields_.end(); ++i){
    bbox.AddField(i->first, (i->second).to(device));
  }
  return bbox;
}

BoxList BoxList::operator[](torch::Tensor item){
  assert(item.sizes().size() == 1);
  item = item.to(bbox_.device());
  
  if(item.dtype().Match<bool>()){
    BoxList bbox = BoxList(bbox_.masked_select(item.unsqueeze(1)).reshape({-1, 4}), size_, mode_);
    for(auto i = extra_fields_.begin(); i != extra_fields_.end(); ++i){
      auto size_vector = (i->second).sizes().vec();
      size_vector[0] = -1;
      while(size_vector.size() != item.sizes().size())
        item.unsqueeze_(-1);
      bbox.AddField(i->first, (i->second).masked_select(item).reshape(torch::IntArrayRef(size_vector)));
    }
    if(rles_.size()){
      std::vector<coco::RLEstr> tmp;
      assert(rles_.size() == item.size(0));
      for(int i = 0; i < item.size(0); ++i){
        if(item.select(0, i).item<int>())
          tmp.push_back(rles_[i]);
      }
      bbox.set_rles(tmp);
    }
    if(masks_){
      bbox.set_masks(new SegmentationMask((*masks_)[item]));
    }
    return bbox;
  }
  else{
    //index_select
    BoxList bbox = BoxList(bbox_.index_select(/*dim=*/0, item), size_, mode_);
    for(auto i = extra_fields_.begin(); i != extra_fields_.end(); ++i){
      auto size_vector = (i->second).sizes().vec();
      size_vector[0] = -1;
      bbox.AddField(i->first, (i->second).index_select(/*dim=*/0, item).reshape(torch::IntArrayRef(size_vector)));
    }
    if(rles_.size()){
      std::vector<coco::RLEstr> tmp;
      for(int i = 0; i < item.size(0); ++i){
        tmp.push_back(rles_[item.select(0, i).item<int64_t>()]);
      }
      bbox.set_rles(tmp);
    }
    if(masks_){
      bbox.set_masks(new SegmentationMask((*masks_)[item]));
    }
    return bbox;
  }
}

BoxList BoxList::operator[](const int64_t index){
    BoxList bbox = BoxList(this->bbox_[index].reshape({-1, 4}), this->size_, this->mode_);
    for(auto i = extra_fields_.begin(); i != extra_fields_.end(); ++i){
        auto size_vector = (i->second).sizes().vec();
        size_vector[0] = -1;
        bbox.AddField(i->first, (i->second)[index].reshape(at::IntArrayRef(size_vector)));
    }
    if(rles_.size()){
      bbox.set_rles(std::vector<coco::RLEstr>{rles_[index]});
    }
    if(masks_)
      bbox.set_masks(new SegmentationMask((*masks_)[index]));
    return bbox;
}

int64_t BoxList::Length() const {
    return bbox_.size(0);
}

BoxList BoxList::ClipToImage(const bool remove_empty){
  int TO_REMOVE = 1;
  bbox_.select(/*dim*/1, /*index*/0).clamp_(/*min*/0, /*max*/std::get<0>(size_) - TO_REMOVE);
  bbox_.select(/*dim*/1, /*index*/1).clamp_(/*min*/0, /*max*/std::get<1>(size_) - TO_REMOVE);
  bbox_.select(/*dim*/1, /*index*/2).clamp_(/*min*/0, /*max*/std::get<0>(size_) - TO_REMOVE);
  bbox_.select(/*dim*/1, /*index*/3).clamp_(/*min*/0, /*max*/std::get<1>(size_) - TO_REMOVE);
  if(remove_empty){
    auto keep = (bbox_.select(1, 3) > bbox_.select(1, 1)).__and__((bbox_.select(1, 2) > bbox_.select(1, 0)));

    return (*this)[keep];
  }
  return *this;
}

torch::Tensor BoxList::Area(){
  if(mode_.compare("xyxy") == 0){
    int TO_REMOVE = 1;
    torch::Tensor area = (bbox_.select(1, 2) - bbox_.select(1, 0) + TO_REMOVE) * (bbox_.select(1, 3) - bbox_.select(1, 1) + TO_REMOVE);
    return area;
  }
  else if(mode_.compare("xywh") == 0){
    torch::Tensor area = bbox_.select(1, 2) * bbox_.select(1, 3);
    return area;
  }
  else{
    throw std::invalid_argument("field is not found");
  }
}

BoxList BoxList::CopyWithFields(const std::vector<std::string> fields, const bool skip_missing){
  BoxList bbox = BoxList(bbox_, size_, mode_);
  for(auto i = fields.begin(); i != fields.end(); ++i){
    if(HasField(*i)){
      if((*i).compare("mask") && rles_.size() > 0)
        bbox.AddField("mask", rles_);
      else if((*i).compare("masks") && masks_)
        bbox.AddField("masks", masks_);
      else
        bbox.AddField(*i, GetField(*i));
    }
    else if(!skip_missing){
      throw std::invalid_argument("field is not found");
    }
  }
    
  return bbox;
}

BoxList BoxList::nms(const float nms_thresh, const int max_proposals, const std::string score_field){
  if(nms_thresh <= 0){
    return *this;
  }
  std::string mode = mode_;
  BoxList boxlist = Convert("xyxy");
  torch::Tensor boxes = boxlist.get_bbox();
  torch::Tensor scores = boxlist.GetField(score_field);
  torch::Tensor keep = rcnn::layers::nms(boxes, scores, nms_thresh);
  if(max_proposals > 0)
    keep = keep.slice(/*dim=*/0, /*start=*/0, /*end=*/max_proposals);
  return boxlist[keep].Convert(mode);
}

BoxList BoxList::RemoveSmallBoxes(const int min_size){
  std::vector<torch::Tensor> bbox = Convert("xywh").get_bbox().unbind(/*dim=*/1);
  torch::Tensor keep = (bbox[2] >= min_size).__and__(bbox[3] >= min_size);
  return (*this)[keep];
}

std::ostream& operator << (std::ostream& os, const BoxList& bbox){
  os << "BoxList(";
  os << "num_boxes=" << bbox.Length() << ", ";
  os << "image_width=" << std::get<0>(bbox.get_size()) << ", ";
  os << "image_height=" << std::get<1>(bbox.get_size()) << ", ";
  os << "mode=" << bbox.get_mode() << ")" << std::endl;
  return os;
}

std::map<std::string, torch::Tensor> BoxList::get_extra_fields() const {
  return extra_fields_;
}

std::vector<coco::RLEstr> BoxList::get_rles() const {
  return rles_;
}

SegmentationMask* BoxList::get_masks() const {
  return masks_;
}

std::pair<Width, Height> BoxList::get_size() const {
  return size_;
}

torch::Device BoxList::get_device() const {
  return device_;
}

torch::Tensor BoxList::get_bbox() const {
  return bbox_;
}

std::string BoxList::get_mode() const {
  return mode_;
}

void BoxList::set_size(const std::pair<Width, Height> size){
  size_ = size;
}

void BoxList::set_masks(SegmentationMask* masks){
  if(masks_)
    delete masks_;
  masks_ = masks;
}

void BoxList::set_rles(const std::vector<coco::RLEstr> rles){
  rles_ = rles;
}

void BoxList::set_extra_fields(const std::map<std::string, torch::Tensor> fields){
  extra_fields_ = fields;
}

void BoxList::set_bbox(const torch::Tensor bbox){
  bbox_ = bbox;
  device_ = bbox.device();
}

void BoxList::set_mode(const std::string mode){
  mode_ = mode;
}

void BoxList::set_mode(const char* mode){
  mode_ = mode;
}

}//structures
}//rcnn