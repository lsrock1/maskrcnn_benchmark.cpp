#include "datasets/coco_datasets.h"
#include <algorithm>
#include <iostream>

#include <bounding_box.h>
#include <segmentation_mask.h>


namespace rcnn{
namespace data{

bool _has_only_empty_bbox(std::vector<coco::Annotation> anno){
  for(auto& i : anno){
    if(i.bbox[2] > 1 && i.bbox[3] > 1)
      return false;
  }
  return true;
}

bool has_valid_annotation(std::vector<coco::Annotation> anno){
  if(anno.size() == 0)
    return false;

  if(_has_only_empty_bbox(anno))
    return false;

  return true;
}

// template<typename Self>
// torch::optional<size_t> RCNNDataset<Self>::size() const{
//   assert(false);
//   return 0;
// }

// template<typename Self>
// torch::data::Example<torch::Tensor, RCNNData> RCNNDataset<Self>::get(size_t index){
//   assert(false);
//   return torch::data::Example<torch::Tensor, RCNNData> ();
// }

COCODataset::COCODataset(std::string annFile, std::string root, bool remove_images_without_annotations) :coco_detection(root, annFile){
  std::sort(coco_detection.ids_.begin(), coco_detection.ids_.end());
  if(remove_images_without_annotations){
    std::vector<int> ids;
    for(auto& i : coco_detection.ids_){
      auto ann_ids = coco_detection.coco_.GetAnnIds(std::vector<int> {i});
      std::vector<coco::Annotation> anno = coco_detection.coco_.LoadAnns(ann_ids);
      if(has_valid_annotation(anno))
        ids.push_back(i);
    }
    coco_detection.ids_ = ids;
  }

  for(auto& cat : coco_detection.coco_.cats)
    categories[cat.second.id] = cat.second.name;
  
  std::vector<int> catIds = coco_detection.coco_.GetCatIds();
  for(int i = 0; i < catIds.size(); ++i)
    json_category_id_to_contiguous_id[catIds[i]] = i + 1;

  for(auto i = json_category_id_to_contiguous_id.begin(); i != json_category_id_to_contiguous_id.end(); ++i)
    contiguous_category_id_to_json_id[i->second] = i->first;

  for(int i = 0; i < coco_detection.ids_.size(); ++i)
    id_to_img_map[i] = coco_detection.ids_[i];
}

torch::data::Example<cv::Mat, RCNNData> COCODataset::get(size_t idx){
  auto coco_data = coco_detection.get(idx);
  cv::Mat img = coco_data.data;
  std::vector<coco::Annotation> anno = coco_data.target;
  for(auto ann = anno.begin(); ann != anno.end();){
    if(ann->iscrowd)
      anno.erase(ann);
    else
      ann++;
  }

  std::vector<std::vector<float>> boxes;
  for(auto& obj : anno)
    boxes.push_back(obj.bbox);
  torch::Tensor boxes_tensor = torch::zeros({static_cast<int64_t>(boxes.size()) * 4}).to(torch::kF32);
  int64_t index = 0;
  for(auto& box : boxes){
    for(auto& coord : box){
      boxes_tensor[index] = coord;
      index++;
    }
  }

  boxes_tensor = boxes_tensor.reshape({-1, 4});
  rcnn::structures::BoxList target{boxes_tensor, std::make_pair(static_cast<int64_t>(img.cols), static_cast<int64_t>(img.rows)), "xywh"};
  target = target.Convert("xyxy");
  
  torch::Tensor classes = torch::zeros({static_cast<int64_t>(anno.size())}).to(torch::kF32);
  for(int i = 0; i < anno.size(); ++i){
    classes[i] = json_category_id_to_contiguous_id[anno[i].category_id];
  }
  
  target.AddField("labels", classes);
  std::vector<std::vector<std::vector<double>>> polys;
  for(auto& obj : anno)
    polys.push_back(obj.segmentation);
  auto mask = new rcnn::structures::SegmentationMask(polys, std::make_pair(static_cast<int64_t>(img.cols), static_cast<int64_t>(img.rows)), "poly");
  target.AddField("masks", mask);

  target = target.ClipToImage(true);
  RCNNData rcnn_data;
  rcnn_data.idx = idx;
  rcnn_data.target = target;
  torch::data::Example<cv::Mat, RCNNData> value{img, rcnn_data};
  return value;
}

torch::optional<size_t>  COCODataset::size() const{
  return coco_detection.size();
}

coco::Image COCODataset::get_img_info(int64_t index){
  auto img_id = id_to_img_map[index];
  return coco_detection.coco_.imgs[img_id];
}

}//data
}//rcnn