#pragma once
#include "datasets/coco_datasets.h"

#include <bounding_box.h>

#include "rapidjson/document.h"


namespace rcnn{
namespace data{

using namespace rapidjson;

void DoCOCOEvaluation(COCODataset& dataset, 
                 std::vector<rcnn::structures::BoxList>& predictions,
                 bool box_only,
                 std::string output_folder,
                 std::vector<std::string> iou_types);
                 //TODO expected results

void prepare_for_coco_detection(std::vector<rcnn::structures::BoxList>& predictions, COCODataset& dataset); 
void prepare_for_coco_segmentation(std::vector<rcnn::structures::BoxList>& predictions, COCODataset& dataset);

}
}