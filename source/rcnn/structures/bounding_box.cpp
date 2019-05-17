#include "bounding_box.h"
#include "nms.h"
#include <cassert>
#include <algorithm>


namespace rcnn{
namespace structures{

torch::Tensor BoxListIOU(BoxList a, BoxList b){
  assert(a.get_size() == b.get_size());
  int TO_REMOVE = 1;
  a = a.Convert("xyxy");
  b = b.Convert("xyxy");
  torch::Tensor area_a = a.Area();
  torch::Tensor area_b = b.Area();
  torch::Tensor bbox_a = a.get_bbox();
  torch::Tensor bbox_b = b.get_bbox();
  torch::Tensor lt = torch::max(bbox_a.unsqueeze(1).slice(/*dim=*/2, /*start=*/0, /*end=*/2), bbox_b.slice(1, 0, 2));
  torch::Tensor rb = torch::min(bbox_a.unsqueeze(1).slice(/*dim=*/2, /*start=*/2, /*end=*/4), bbox_b.slice(1, 2, 4));

  torch::Tensor wh = (rb - lt + TO_REMOVE).clamp(0);
  torch::Tensor inter = wh.slice(2, 0, 1) * wh.slice(2, 1, 1);
  return inter / (area_a.unsqueeze(1) + area_b - inter);
}

BoxList BoxListCat(std::vector<BoxList> boxlists){
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
    assert(boxlist.get_size() == size && boxlist.get_mode() == mode && fields == compared_field);
  }
  BoxList cat_boxlists = BoxList(torch::cat(cat_bbox, 0), size, mode);
  for(auto& field: fields){
    std::vector<torch::Tensor> cat_field;
    for(auto& boxlist: boxlists){
      cat_field.push_back(boxlist.GetField(field));
    }
    cat_boxlists.AddField(field, torch::cat(cat_field, 0));
    cat_field.clear();
  }
  return cat_boxlists;
}

//size<width, height>
BoxList::BoxList(torch::Tensor bbox, std::pair<Width, Height> image_size, const char* mode)
    : device_(bbox.device()),
      size_(image_size),
      bbox_(bbox),
      mode_(mode){};

BoxList::BoxList(torch::Tensor bbox, std::pair<Width, Height> image_size, std::string mode)
    : device_(bbox.device()),
      size_(image_size),
      bbox_(bbox),
      mode_(mode){};

//only supports tensor field data
void BoxList::AddField(const std::string field_name, torch::Tensor field_data){
    this->extra_fields_[field_name] = field_data;
}

torch::Tensor BoxList::GetField(const std::string field_name){
    return this->extra_fields_.find(field_name)->second;
}

bool BoxList::HasField(const std::string field_name){
    return this->extra_fields_.count(field_name) ? true : false;
}

std::vector<std::string> BoxList::Fields(){
    std::vector<std::string> keys;
    for(auto i = extra_fields_.begin(); i != extra_fields_.end(); i++)
        keys.push_back(i->first);
    return keys;
}

BoxList BoxList::Convert(const std::string mode){
    assert(mode.compare("xyxy") == 0 || mode.compare("xywh") == 0);
    if(this->mode_.compare(mode) != 0){
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
        BoxList bbox = BoxList(bbox_tensor, this->size_, mode);
        bbox.CopyExtraFields(*this);
        return bbox;
    }
    else{
        return *this;
    }
}

std::tuple<XMin, YMin, XMax, YMax> BoxList::SplitIntoXYXY(){
    if(this->mode_.compare("xyxy") == 0){
        std::vector<torch::Tensor> splitted_box_coordinates = this->bbox_.split(1, /*dim=*/-1);
        return std::make_tuple(
            splitted_box_coordinates.at(0),
            splitted_box_coordinates.at(1),
            splitted_box_coordinates.at(2),
            splitted_box_coordinates.at(3));
    }
    else if(mode_.compare("xywh") == 0){
        int TO_REMOVE = 1;
        std::vector<torch::Tensor> splitted_box_coordinates = this->bbox_.split(1, /*dim=*/-1);
        return std::make_tuple(
            splitted_box_coordinates.at(0),
            splitted_box_coordinates.at(1),
            splitted_box_coordinates.at(0) + (splitted_box_coordinates.at(2) - TO_REMOVE).clamp_min(0),
            splitted_box_coordinates.at(1) + (splitted_box_coordinates.at(3) - TO_REMOVE).clamp_min(0)
        );
    }
}

void BoxList::CopyExtraFields(const BoxList bbox){
    if(!bbox.get_extra_fields().empty()){
        auto extra_field_src = bbox.get_extra_fields();
        for(auto i = extra_field_src.begin(); i != extra_field_src.end(); ++i){
            this->extra_fields_[i->first] = i->second;
        }
    }
}

BoxList BoxList::Resize(const std::pair<Width, Height> size){
    //width, height
    std::pair<float, float> ratios = std::make_pair(float(size.first) / float(this->size_.first), float(size.second) / float(this->size_.second));
    torch::Tensor scaled_bbox;
    if(ratios.first == ratios.second){
        float ratio = ratios.first;
        scaled_bbox = this->bbox_ * ratio;
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
    return bbox.Convert(this->get_mode());
}

BoxList BoxList::Transpose(const int method){
    assert(method == BoxList::kFlipLeftRight || method == BoxList::kFlipTopBottom);
    int image_width = this->size_.first, image_height = this->size_.second;
    std::tuple<XMin, YMin, XMax, YMax> splitted_box_coordinates = SplitIntoXYXY();
    XMin transposed_xmin;
    XMax transposed_xmax;
    YMin transposed_ymin;
    YMax transposed_ymax;
    if(method == BoxList::kFlipLeftRight){
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
    BoxList bbox = BoxList(transposed_boxes, this->size_, "xyxy");
    return bbox.Convert(this->mode_);
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
    bbox.CopyExtraFields(*this);
    return bbox.Convert(this->mode_);
}

BoxList BoxList::To(const torch::Device device){
    BoxList bbox = BoxList(this->bbox_.to(device), this->size_, this->mode_);
    for(auto i = this->extra_fields_.begin(); i != this->extra_fields_.end(); ++i){
        bbox.AddField(i->first, (i->second).to(device));
    }
    return bbox;
}

BoxList BoxList::operator[](torch::Tensor item){
    assert(item.sizes().size() == 1);
    if(item.dtype() == torch::kByte){
      BoxList bbox = BoxList(this->bbox_.masked_select(item.unsqueeze(1)).reshape({-1, 4}), this->size_, this->mode_);
      for(auto i = this->extra_fields_.begin(); i != this->extra_fields_.end(); ++i){
        auto size_vector = (i->second).sizes().vec();
        size_vector[0] = -1;
        while(size_vector.size() != item.sizes().size())
          item.unsqueeze_(-1);
        bbox.AddField(i->first, (i->second).masked_select(item).reshape(at::IntArrayRef(size_vector)));
      }
      return bbox;
    }
    else{
      //index_select
      BoxList bbox = BoxList(this->bbox_.index_select(/*dim=*/0, item), this->size_, this->mode_);
      for(auto i = this->extra_fields_.begin(); i != this->extra_fields_.end(); ++i){
        auto size_vector = (i->second).sizes().vec();
        size_vector[0] = -1;
        bbox.AddField(i->first, (i->second).index_select(/*dim=*/0, item).reshape(at::IntArrayRef(size_vector)));
      }
      return bbox;
    }
}

BoxList BoxList::operator[](const int64_t index){
    BoxList bbox = BoxList(this->bbox_[index].reshape({-1, 4}), this->size_, this->mode_);
    for(auto i = this->extra_fields_.begin(); i != this->extra_fields_.end(); ++i){
        auto size_vector = (i->second).sizes().vec();
        size_vector[0] = -1;
        bbox.AddField(i->first, (i->second)[index].reshape(at::IntArrayRef(size_vector)));
    }
    return bbox;
}

int64_t BoxList::Length() const {
    return this->bbox_.size(0);
}

BoxList BoxList::ClipToImage(const bool remove_empty){
    int TO_REMOVE = 1;
    torch::Tensor bbox_tensor = torch::cat({
        this->bbox_.narrow(/*dim*/1, /*start*/0, /*length*/1).clamp_(/*min*/0, /*max*/std::get<0>(this->size_) - TO_REMOVE),
        this->bbox_.narrow(/*dim*/1, /*start*/1, /*length*/1).clamp_(/*min*/0, /*max*/std::get<1>(this->size_) - TO_REMOVE),
        this->bbox_.narrow(/*dim*/1, /*start*/2, /*length*/1).clamp_(/*min*/0, /*max*/std::get<0>(this->size_) - TO_REMOVE),
        this->bbox_.narrow(/*dim*/1, /*start*/3, /*length*/1).clamp_(/*min*/0, /*max*/std::get<1>(this->size_) - TO_REMOVE)}, 1);
    if(remove_empty){
        auto keep = (bbox_tensor.narrow(1, 3, 1) > bbox_tensor.narrow(1, 1, 1)).__and__((bbox_tensor.narrow(1, 2, 1) > bbox_tensor.narrow(1, 0, 1)));
        return (*this)[keep];
    }
    return *this;
}

torch::Tensor BoxList::Area(){
    if(this->mode_.compare("xyxy") == 0){
        int TO_REMOVE = 1;
        torch::Tensor area = (this->bbox_.narrow(1, 2, 1) - this->bbox_.narrow(1, 0, 1) + TO_REMOVE) * (this->bbox_.narrow(1, 3, 1) - this->bbox_.narrow(1, 1, 1) + TO_REMOVE);
        return area;
    }
    else if(this->mode_.compare("xywh") == 0){
        torch::Tensor area = this->bbox_.narrow(1, 2, 1) * this->bbox_.narrow(1, 3, 1);
        return area;
    }
    else{
        throw std::invalid_argument("field is not found");
    }
}

BoxList BoxList::CopyWithFields(const std::vector<std::string> fields, const bool skip_missing){
    BoxList bbox = BoxList(this->bbox_, this->size_, this->mode_);
    for(auto i = fields.begin(); i != fields.end(); ++i){
        if(this->HasField(*i)){
            bbox.AddField(*i, this->GetField(*i));
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
  BoxList boxlist = this->Convert("xyxy");
  torch::Tensor boxes = boxlist.get_bbox();
  torch::Tensor scores = boxlist.GetField(score_field);
  torch::Tensor keep = rcnn::layers::nms(boxes, scores, nms_thresh);
  if(max_proposals > 0 && keep.size(0) > max_proposals)
    keep = keep.narrow(/*dim=*/0, /*start=*/0, /*end=*/max_proposals);
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
    return this->extra_fields_;
}

std::pair<Width, Height> BoxList::get_size() const {
    return this->size_;
}

torch::Device BoxList::get_device() const {
    return this->device_;
}

torch::Tensor BoxList::get_bbox() const {
    return this->bbox_;
}

std::string BoxList::get_mode() const {
    return this->mode_;
}

void BoxList::set_size(const std::pair<Width, Height> size){
    this->size_ = size;
}

void BoxList::set_extra_fields(const std::map<std::string, torch::Tensor> fields){
    this->extra_fields_ = fields;
}

void BoxList::set_bbox(const torch::Tensor bbox){
    this->bbox_ = bbox;
    this->device_ = bbox.device();
}

void BoxList::set_mode(const std::string mode){
    this->mode_ = mode;
}

void BoxList::set_mode(const char* mode){
    this->mode_ = mode;
}
}//structures
}//rcnn