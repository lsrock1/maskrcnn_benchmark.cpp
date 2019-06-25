#include "paths_catalog.h"
#include <cassert>
#include "defaults.h"


namespace rcnn{
namespace config{

const std::string DatasetCatalog::DATA_DIR = "datasets";
const std::map<std::string, args> DatasetCatalog::DATASETS{
  {"coco_2017_train", 
    args{{"img_dir", "coco/train2017"}, {"ann_file", "coco/annotations/instances_train2017.json"}}},
  {"coco_2017_val", 
    args{{"img_dir", "coco/val2017"}, {"ann_file", "coco/annotations/instances_val2017.json"}}},
  {"coco_2014_train", 
    args{{"img_dir", "coco/train2014"}, {"ann_file", "coco/annotations/instances_train2014.json"}}},
  {"coco_2014_val",
    args{{"img_dir", "coco/val2014"}, {"ann_file", "coco/annotations/instances_val2014.json"}}},
  {"coco_2014_minival",
    args{{"img_dir", "coco/val2014"}, {"ann_file", "coco/annotations/instances_valminusminival2014.json"}}}
};

std::tuple<std::string, std::string, std::string> DatasetCatalog::operator[](std::string name){
  if(name.find("coco") != std::string::npos){
    return std::make_tuple("COCODataset", DATA_DIR + "/" + DATASETS.at(name).at("img_dir"), DATA_DIR + "/" + DATASETS.at(name).at("ann_file"));
  }
  //no voc
  assert(false);
}

const std::string ModelCatalog::S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron";
const std::string ModelCatalog::C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl";
const std::map<std::string, std::string> ModelCatalog::C2_IMAGENET_MODELS{
  {"MSRA/R-50", "ImageNetPretrained/MSRA/R-50.pkl"},
  {"MSRA/R-50-GN", "ImageNetPretrained/47261647/R-50-GN.pkl"},
  {"MSRA/R-101", "ImageNetPretrained/MSRA/R-101.pkl"},
  {"MSRA/R-101-GN", "ImageNetPretrained/47592356/R-101-GN.pkl"},
  {"FAIR/20171220/X-101-32x8d", "ImageNetPretrained/20171220/X-101-32x8d.pkl"}
};
const std::map<std::string, std::string> ModelCatalog::C2_DETECTRON_MODELS{
  {"35857197/e2e_faster_rcnn_R-50-C4_1x", "01_33_49.iAX0mXvW"},
  {"35857345/e2e_faster_rcnn_R-50-FPN_1x", "01_36_30.cUF7QR7I"},
  {"35857890/e2e_faster_rcnn_R-101-FPN_1x", "01_38_50.sNxI7sX7"},
  {"36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x", "06_31_39.5MIHi1fZ"},
  {"35858791/e2e_mask_rcnn_R-50-C4_1x", "01_45_57.ZgkA7hPB"},
  {"35858933/e2e_mask_rcnn_R-50-FPN_1x", "01_48_14.DzEQe4wC"},
  {"35861795/e2e_mask_rcnn_R-101-FPN_1x", "02_31_37.KqyEK4tT"},
  {"36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x", "06_35_59.RZotkLKI"},
  {"37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x", "09_35_36.8pzTQKYK"},
        //keypoints
  {"37697547/e2e_keypoint_rcnn_R-50-FPN_1x", "08_42_54.kdzV35ao"}
};

std::string ModelCatalog::get(std::string name){
  if(name.find("Caffe2Detectron/COCO") == 0)
    return get_c2_detectron_12_2017_baselines(name);
  if(name.find("ImageNetPretrained") == 0)
    return get_c2_imagenet_pretrained(name);
  assert(false);
}

std::string ModelCatalog::get_c2_imagenet_pretrained(std::string name){
  std::string prefix = "S3_C2_DETECTRON_URL";
  std::string image_net_prefix("ImageNetPretrained/");
  name = name.substr(image_net_prefix.size());
  name = C2_IMAGENET_MODELS.at(name);
  return prefix + "/" + name;
}

std::string ModelCatalog::get_c2_detectron_12_2017_baselines(std::string name){
  std::string prefix = "S3_C2_DETECTRON_URL";
  std::string dataset_tag = name.find("keypoint") != std::string::npos ? "keypoints_" : "";
  const std::string suffix = C2_DETECTRON_SUFFIX.at(0) + dataset_tag + C2_DETECTRON_SUFFIX.at(1) + dataset_tag + C2_DETECTRON_SUFFIX.at(2);
  std::string caffe_prefix = "Caffe2Detectron/COCO/";
  name = name.substr(caffe_prefix.size());
  
  std::string model_id = name.substr(0, name.find("/"));
  std::string model_name = name.substr(name.find("/")+1);
  model_name = model_name + ".yaml";
  const std::string signature = C2_DETECTRON_MODELS.at(name);
  std::string unique_name = model_name + "." + signature;
  return prefix + "/" + model_id + "/12_2017_baselines/" + unique_name + "/" + suffix;
}

}
}