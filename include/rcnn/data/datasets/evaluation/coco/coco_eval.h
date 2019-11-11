#pragma once
#include "datasets/coco_datasets.h"

#include <bounding_box.h>

#include "rapidjson/document.h"

#include <set>
#include <map>

using namespace rapidjson;

namespace rcnn {
namespace data {

void DoCOCOEvaluation(COCODataset& dataset, 
                      std::map<int64_t, rcnn::structures::BoxList>& predictions,
                      const std::string& output_folder,
                      std::set<std::string> iou_types,
                      const std::string& ann_file);
                      //TODO expected results

void prepare_for_coco_detection(const std::string& output_folder, std::map<int64_t, rcnn::structures::BoxList>& predictions, COCODataset& dataset); 
// void prepare_for_coco_segmentation(std::map<int64_t, rcnn::structures::BoxList>& predictions, COCODataset& dataset);

} // namespace data
} // namespace rcnn
