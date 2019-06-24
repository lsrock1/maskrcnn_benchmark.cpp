#include "coco_detection.h"


namespace rcnn{
namespace data{

COCODetection::COCODetection(std::string root, std::string annFile)
                            :root_(root),
                             coco_(coco::COCO(annFile))
{
  ids_.reserve(coco_.imgs.size());
  for(auto& img : coco_.imgs)
    ids_.push_back(img.first);  
}

torch::data::Example<cv::Mat, std::vector<coco::Annotation>> COCODetection::get(size_t index){
  int img_id = ids_.at(index);
  std::vector<int64_t> ann_ids = coco_.GetAnnIds(std::vector<int>{img_id});
  std::vector<coco::Annotation> target = coco_.LoadAnns(ann_ids);
  std::string path(coco_.LoadImgs(std::vector<int>{img_id})[0].file_name);
  cv::Mat img = cv::imread(root_ + "/" + path, cv::IMREAD_COLOR);
  

  torch::data::Example<cv::Mat, std::vector<coco::Annotation>> value{img, target};
  return value;
}

torch::optional<size_t> COCODetection::size() const{
  return ids_.size();
}

std::ostream& operator << (std::ostream& os, const COCODetection& bml){
  os << "Dataset COCODetection\n";
  os << "   Number of datapoints: " << bml.size() << "\n";
  os << "   Root Location: " << bml.root_ << "\n";
  return os;
}

}//data
}//rcnn