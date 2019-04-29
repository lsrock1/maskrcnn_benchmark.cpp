#include "bounding_box.h"
#include <cassert>
#include <ATen/ATen.h>
#include <stdexcept>


using namespace std;
namespace rcnn{
namespace structures{

//size<width, height>
BoxList::BoxList(torch::Tensor bbox, pair<BoxList::Width, BoxList::Height> image_size, const char* mode)
    : device_(bbox.device()),
      size_(image_size),
      bbox_(bbox),
      mode_(mode){};

BoxList::BoxList(torch::Tensor bbox, pair<BoxList::Width, BoxList::Height> image_size, string mode)
    : device_(bbox.device()),
      size_(image_size),
      bbox_(bbox),
      mode_(mode){};

//only supports tensor field data
void BoxList::AddField(const string field_name, torch::Tensor field_data){
    this->extra_fields_[field_name] = field_data;
}

torch::Tensor BoxList::GetField(const string field_name){
    return this->extra_fields_.find(field_name)->second;
}

bool BoxList::HasField(const string field_name){
    return this->extra_fields_.count(field_name) ? true : false;
}

vector<string> BoxList::Fields(){
    vector<string> keys;
    for(auto i = extra_fields_.begin(); i != extra_fields_.end(); i++)
        keys.push_back(i->first);
    return keys;
}

BoxList BoxList::Convert(const string mode){
    assert(mode.compare("xyxy") == 0 || mode.compare("xywh") == 0);
    if(this->mode_.compare(mode) != 0){
        tuple<BoxList::XMin, BoxList::YMin, BoxList::XMax, BoxList::YMax> splitted_box_coordinates = SplitIntoXYXY();
        torch::Tensor bbox_tensor;
        if(mode.compare("xyxy") == 0){
            bbox_tensor = torch::cat({
                get<0>(splitted_box_coordinates),
                get<1>(splitted_box_coordinates),
                get<2>(splitted_box_coordinates),
                get<3>(splitted_box_coordinates),
            }, -1);
        }
        else{
            int TO_REMOVE = 1;
            bbox_tensor = torch::cat({
                get<0>(splitted_box_coordinates),
                get<1>(splitted_box_coordinates),
                get<2>(splitted_box_coordinates) - get<0>(splitted_box_coordinates) + TO_REMOVE,
                get<3>(splitted_box_coordinates) - get<1>(splitted_box_coordinates) + TO_REMOVE
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

tuple<BoxList::XMin, BoxList::YMin, BoxList::XMax, BoxList::YMax> BoxList::SplitIntoXYXY(){
    if(this->mode_.compare("xyxy") == 0){
        vector<torch::Tensor> splitted_box_coordinates = this->bbox_.split(1, /*dim=*/-1);
        return make_tuple(
            splitted_box_coordinates.at(0),
            splitted_box_coordinates.at(1),
            splitted_box_coordinates.at(2),
            splitted_box_coordinates.at(3));
    }
    else if(mode_.compare("xywh") == 0){
        int TO_REMOVE = 1;
        vector<torch::Tensor> splitted_box_coordinates = this->bbox_.split(1, /*dim=*/-1);
        return make_tuple(
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

BoxList BoxList::Resize(const pair<BoxList::Width, BoxList::Height> size){
    //width, height
    pair<float, float> ratios = make_pair(float(size.first) / float(this->size_.first), float(size.second) / float(this->size_.second));
    torch::Tensor scaled_bbox;
    if(ratios.first == ratios.second){
        float ratio = ratios.first;
        scaled_bbox = this->bbox_ * ratio;
    }
    else{
        float ratio_width = ratios.first, ratio_height = ratios.second;
        tuple<BoxList::XMin, BoxList::YMin, BoxList::XMax, BoxList::YMax> splitted_box_coordinates = SplitIntoXYXY();
        BoxList::XMin scaled_xmin = get<0>(splitted_box_coordinates) * ratio_width;
        BoxList::XMax scaled_xmax = get<2>(splitted_box_coordinates) * ratio_width;
        BoxList::YMin scaled_ymin = get<1>(splitted_box_coordinates) * ratio_height;
        BoxList::YMax scaled_ymax = get<3>(splitted_box_coordinates) * ratio_height;
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
    tuple<BoxList::XMin, BoxList::YMin, BoxList::XMax, BoxList::YMax> splitted_box_coordinates = SplitIntoXYXY();
    BoxList::XMin transposed_xmin;
    BoxList::XMax transposed_xmax;
    BoxList::YMin transposed_ymin;
    BoxList::YMax transposed_ymax;
    if(method == BoxList::kFlipLeftRight){
        int TO_REMOVE = 1;
        transposed_xmin = image_width - get<2>(splitted_box_coordinates) - TO_REMOVE;
        transposed_xmax = image_width - get<0>(splitted_box_coordinates) - TO_REMOVE;
        transposed_ymin = get<1>(splitted_box_coordinates);
        transposed_ymax = get<3>(splitted_box_coordinates);
    }
    else{
        transposed_xmin = get<0>(splitted_box_coordinates);
        transposed_xmax = get<2>(splitted_box_coordinates);
        transposed_ymin = image_height - get<3>(splitted_box_coordinates);
        transposed_ymax = image_height - get<1>(splitted_box_coordinates);
    }

    torch::Tensor transposed_boxes = torch::cat({
        transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax
    }, -1);
    BoxList bbox = BoxList(transposed_boxes, this->size_, "xyxy");
    return bbox.Convert(this->mode_);
}

BoxList BoxList::Crop(const tuple<int64_t, int64_t, int64_t, int64_t> box){
    tuple<BoxList::XMin, BoxList::YMin, BoxList::XMax, BoxList::YMax> splitted_box_coordinates = SplitIntoXYXY();
    int64_t width = get<2>(box) - get<0>(box), height = get<3>(box) - get<1>(box);
    BoxList::XMin cropped_xmin = (get<0>(splitted_box_coordinates) - get<0>(box)).clamp(/*min*/0, /*max*/width);
    BoxList::YMin cropped_ymin = (get<1>(splitted_box_coordinates) - get<1>(box)).clamp(/*min*/0, /*max*/height);
    BoxList::XMax cropped_xmax = (get<2>(splitted_box_coordinates) - get<0>(box)).clamp(/*min*/0, /*max*/width);
    BoxList::YMax cropped_ymax = (get<3>(splitted_box_coordinates) - get<1>(box)).clamp(/*min*/0, /*max*/height);

    torch::Tensor cropped_box = torch::cat(
        {cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax}, -1
    );
    BoxList bbox = BoxList(cropped_box, make_pair(width, height), "xyxy");
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
    assert(item.sizes().size() == 1 && item.size(0) == this->bbox_.size(0));
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
        this->bbox_.narrow(/*dim*/1, /*start*/0, /*length*/1).clamp_(/*min*/0, /*max*/get<0>(this->size_) - TO_REMOVE),
        this->bbox_.narrow(/*dim*/1, /*start*/1, /*length*/1).clamp_(/*min*/0, /*max*/get<1>(this->size_) - TO_REMOVE),
        this->bbox_.narrow(/*dim*/1, /*start*/2, /*length*/1).clamp_(/*min*/0, /*max*/get<0>(this->size_) - TO_REMOVE),
        this->bbox_.narrow(/*dim*/1, /*start*/3, /*length*/1).clamp_(/*min*/0, /*max*/get<1>(this->size_) - TO_REMOVE)}, 1);
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
        throw invalid_argument("field is not found");
    }
}

BoxList BoxList::CopyWithFields(const vector<string> fields, const bool skip_missing){
    BoxList bbox = BoxList(this->bbox_, this->size_, this->mode_);
    for(auto i = fields.begin(); i != fields.end(); ++i){
        if(this->HasField(*i)){
            bbox.AddField(*i, this->GetField(*i));
        }
        else if(!skip_missing){
            throw invalid_argument("field is not found");
        }
    }
    return bbox;
}

ostream& operator << (ostream& os, const BoxList& bbox){
    os << "BoxList(";
    os << "num_boxes=" << bbox.Length() << ", ";
    os << "image_width=" << get<0>(bbox.get_size()) << ", ";
    os << "image_height=" << get<1>(bbox.get_size()) << ", ";
    os << "mode=" << bbox.get_mode() << ")" << endl;
    return os;
}

map<string, torch::Tensor> BoxList::get_extra_fields() const {
    return this->extra_fields_;
}

pair<BoxList::Width, BoxList::Height> BoxList::get_size() const {
    return this->size_;
}

torch::Device BoxList::get_device() const {
    return this->device_;
}

torch::Tensor BoxList::get_bbox() const {
    return this->bbox_;
}

string BoxList::get_mode() const {
    return this->mode_;
}

void BoxList::set_size(const pair<BoxList::Width, BoxList::Height> size){
    this->size_ = size;
}

void BoxList::set_extra_fields(const map<string, torch::Tensor> fields){
    this->extra_fields_ = fields;
}

void BoxList::set_bbox(const torch::Tensor bbox){
    this->bbox_ = bbox;
    this->device_ = bbox.device();
}

void BoxList::set_mode(const string mode){
    this->mode_ = mode;
}

void BoxList::set_mode(const char* mode){
    this->mode_ = mode;
}
}//structures
}//rcnn