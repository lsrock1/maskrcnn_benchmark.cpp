#include <structures/bounding_box.h>
#include <cassert>

using namespace std;

BoxList::BoxList(torch::Tensor bbox, pair<int64_t, int64_t> image_size, string mode)
    : device(bbox.device()),
      size(image_size),
      bbox(bbox),
      mode(mode){};

//only supports tensor field data
void BoxList::add_field(const string field_name, torch::Tensor field_data){
    extra_fields.insert(make_pair(field_name, field_data));
}

torch::Tensor BoxList::get_field(const string field_name){
    return extra_fields.find(field_name)->second;
}

bool BoxList::has_field(const string field_name){
    return extra_fields.count(field_name) ? true : false;
}

vector<string> BoxList::fields(){
    vector<string> keys;
    for(auto i = extra_fields.begin(); i != extra_fields.end(); i++)
        keys.push_back(i->first);
    return keys;
}

BoxList& BoxList::convert(const string mode){
    assert(mode.compare("xyxy") == 0 || mode.compare("xywh") == 0);
    if(self.mode.compare(mode) == 0){
        return self;
    }
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> splitted_box_coordinates = _split_into_xyxy();
    if(mode.compare("xyxy") == 0){
        torch::Tensor bbox_tensor = torch::cat({
            get<0>(splitted_box_coordinates),
            get<1>(splitted_box_coordinates),
            get<2>(splitted_box_coordinates),
            get<3>(splitted_box_coordinates),
        }, -1);
        BoxList bbox = BoxList(bbox_tensor, size, mode);
    }
    else{
        int TO_REMOVE = 1;
        torch::Tensor bbox_tensor = torch::cat({
            get<0>(splitted_box_coordinates),
            get<1>(splitted_box_coordinates),
            get<2>(splitted_box_coordinates) - get<0>(splitted_box_coordinates) + TO_REMOVE,
            get<3>(splitted_box_coordinates) - get<1>(splitted_box_coordinates) + TO_REMOVE
        }, -1);
        BoxList bbox = BoxList(bbox_tensor, size, mode);
    }
    bbox._copy_extra_fields(self);
    return bbox
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BoxList::_split_into_xyxy(){
    if(mode.compare("xyxy") == 0){
        vector<torch::Tensor> splitted_box_coordinates = bbox.split(1, /*dim=*/-1);
        return make_tuple(
            splitted_box_coordinates.at(0),
            splitted_box_coordinates.at(1),
            splitted_box_coordinates.at(2),
            splitted_box_coordinates.at(3));
    }
    else if(mode.compare("xywh") == 0){
        int TO_REMOVE = 1;
        vector<torch::Tensor> splitted_box_coordinates = bbox.split(1, /*dim=*/-1);
        return make_tuple(
            splitted_box_coordinates.at(0),
            splitted_box_coordinates.at(1),
            splitted_box_coordinates.at(0) + (splitted_box_coordinates.at(2) - TO_REMOVE).clamp_min(0),
            splitted_box_coordinates.at(1) + (splitted_box_coordinates.at(3) - TO_REMOVE).clamp_min(0)
        );
    }
}

void BoxList::_copy_extra_fields(BoxList bbox){
    for(auto i = bbox.extra_fields.begin(); i != bbox.extra_fields.end(); ++i){
        self.extra_fields.insert(make_pair(i->first, i->second));
    }
}

BoxList& BoxList::resize(const pair<int64_t, int64_t> size){
    //width, height
    pair<float, float> ratios = make_pair(float(size.first) / float(self.size.first), float(size.size.second) / float(self.size.second));
    if(ratios.first == ratios.second){
        float ratio = ratios.first;
        torch:Tensor scaled_bbox = box * ratio;
        BoxList bbox = BoxList(scaled_bbox, size, mode);
        // for(auto i = extra_fields.begin(); i != extra_fields.end(); ++i){
        // only tensor type extra field supports
        // }
        return bbox
    }
    else{
        float ratio_width = ratios.first, ratio_height = ratios.second;
        tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> splitted_box_coordinates _split_into_xyxy();
        torch::Tensor scaled_xmin = get<0>(splitted_box_coordinates) * ratio_width;
        torch::Tensor scaled_xmax = get<2>(splitted_box_coordinates) * ratio_width;
        torch::Tensor scaled_ymin = get<1>(splitted_box_coordinates) * ratio_height;
        torch::Tensor scaled_ymax = get<3>(splitted_box_coordinates) * ratio_height;
        torch::Tensor scaled_bbox = torch::cat({
            scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax
        }, -1);
        BoxList bbox = BoxList(scaled_bbox, size, "xyxy");
        return bbox.convert(mode);
    }
}

BoxList& BoxList::transpose(const int method){
    assert(method == BoxList.FLIP_LEFT_RIGHT || method == BoxList.FLIP_TOP_BOTTOM);
    int image_width = size.first, image_height = size.second;
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> splitted_box_coordinates = _split_into_xyxy();
    
    if(method == BoxList.FLIP_LEFT_RIGHT){
        int TO_REMOVE = 1;
        torch::Tensor transposed_xmin = image_width - get<2>(splitted_box_coordinates) - TO_REMOVE;
        torch::Tensor transposed_xmax = image_width - get<0>(splitted_box_coordinates) - TO_REMOVE;
        torch::Tensor transposed_ymin = get<1>(splitted_box_coordinates);
        torch::Tensor transposed_ymax = get<3>(splitted_box_coordinates);
    }
    else{
        torch::Tensor transposed_xmin = get<0>(splitted_box_coordinates);
        torch::Tensor transposed_xmax = get<2>(splitted_box_coordinates);
        torch::Tensor transposed_ymin = image_height - get<3>(splitted_box_coordinates);
        torch::Tensor transposed_ymax = image_height - get<1>(splitted_box_coordinates);
    }

    transposed_boxes = torch::cat({
        transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax
    }, -1);
    return BoxList(transposed_boxes, size, "xyxy").convert(mode);
}

BoxList& BoxList::crop(const tuple<int64_t, int64_t, int64_t, int64_t> box){
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> splitted_box_coordinates = _split_into_xyxy();
    int64_t width = get<2>(box) - get<0>(box), height = get<3>(box) - get<1>(box);
    torch::Tensor cropped_xmin = (get<0>(splitted_box_coordinates) - get<0>(box)).clamp(/*min*/0, /*max*/w);
    torch::Tensor cropped_ymin = (get<1>(splitted_box_coordinates) - get<1>(box)).clamp(/*min*/0, /*max*/h);
    torch::Tensor cropped_xmax = (get<2>(splitted_box_coordinates) - get<0>(box)).clamp(/*min*/0, /*max*/w);
    torch::Tensor cropped_ymax = (get<3>(splitted_box_coordinates) - get<1>(box)).clamp(/*min*/0, /*max*/h);

    torch::Tensor cropped_box = torch::cat(
        {cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax}, -1
    );
    BoxList = BoxList(cropped_box, make_pair(width, height), "xyxy");
    return bbox.convert(mode);
}

void BoxList::to(const torch::Device device){
    BoxList bbox = BoxList(self.bbox.to(device), size, mode);
    for(auto i = extra_fields.begin(); i != extra_fields.end(); ++i){
        bbox.add_field(i->first, (i->second).to(device));
    }
    return bbox;
}

BoxList& operator[](const torch::Tensor item){
    assert(item.sizes().size() == 1);
    box.masked_select
}