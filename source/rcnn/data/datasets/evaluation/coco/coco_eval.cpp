#include "datasets/evaluation/coco/coco_eval.h"

#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <fstream>

#include <torch/torch.h>
#include <coco.h>
#include <iostream>


namespace rcnn{
namespace data{

using namespace rapidjson;

void DoCOCOEvaluation(COCODataset& dataset, 
                 std::map<int64_t, rcnn::structures::BoxList>& predictions,
                 std::string output_folder,
                 std::set<std::string> iou_types){
  std::cout << "Preparing results for COCO format\n";

  if(iou_types.count("bbox")){
    std::cout << "Preparing bbox results\n";
    prepare_for_coco_detection(output_folder, predictions, dataset);
  }

  std::cout << "Evaluation is not implemented!\n";
}
                 //TODO expected results

void prepare_for_coco_detection(std::string output_folder, std::map<int64_t, rcnn::structures::BoxList>& predictions, COCODataset& dataset){
  Document coco_results;
  auto& a = coco_results.GetAllocator();
  coco_results.SetArray();
  int64_t original_id, image_width, image_height, image_id;
  coco::Image image_info;
  rcnn::structures::BoxList prediction;
  torch::Tensor bboxes, scores, labels;
  for(auto prediction_set = predictions.begin(); prediction_set != predictions.end(); ++prediction_set){
    image_id = prediction_set->first;
    original_id = dataset.id_to_img_map[image_id];
    prediction = prediction_set->second;
    if(prediction.Length() == 0)
      continue;
    image_info = dataset.get_img_info(image_id);
    image_height = image_info.height;
    image_width = image_info.width;
    prediction = prediction.Resize(std::make_pair(image_width, image_height));
    prediction = prediction.Convert("xywh");

    bboxes = prediction.get_bbox();
    //bboxes = bboxes.reshape({-1});
    scores = prediction.GetField("scores");
    labels = prediction.GetField("labels");

    for(int i = 0; i < scores.size(0); ++i){
      Value node(kObjectType);
      node.AddMember("image_id", original_id, a);
      node.AddMember("score", scores[i].item<double>(), a);
      Value box(kArrayType);
      box.PushBack(bboxes[i][0].item<float>(), a)
         .PushBack(bboxes[i][1].item<float>(), a)
         .PushBack(bboxes[i][2].item<float>(), a)
         .PushBack(bboxes[i][3].item<float>(), a);
      node.AddMember("bbox", box, a);
      node.AddMember("category_id", dataset.contiguous_category_id_to_json_id[labels[i].item<int>()], a);
      coco_results.PushBack(node, a);
    }
  }
 
  std::ofstream ofs(output_folder + "/bbox.json");
  OStreamWrapper osw(ofs);
  Writer<OStreamWrapper> writer(osw);
  coco_results.Accept(writer);
}

void prepare_for_coco_segmentation(std::string output_folder, std::vector<rcnn::structures::BoxList>& predictions, COCODataset& dataset){
  
}

}
}