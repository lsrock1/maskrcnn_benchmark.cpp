#pragma once
#include "datasets/coco_datasets.h"

#include <bounding_box.h>

#include "rapidjson/document.h"

#include <set>
#include <map>


namespace rcnn{
namespace data{

using namespace rapidjson;

void DoCOCOEvaluation(COCODataset& dataset, 
                 std::map<int64_t, rcnn::structures::BoxList>& predictions,
                 std::string output_folder,
                 std::set<std::string> iou_types,
                 std::string ann_file);
                 //TODO expected results

void prepare_for_coco_detection(std::string output_folder, std::map<int64_t, rcnn::structures::BoxList>& predictions, COCODataset& dataset); 
void prepare_for_coco_segmentation(std::map<int64_t, rcnn::structures::BoxList>& predictions, COCODataset& dataset);

}
}