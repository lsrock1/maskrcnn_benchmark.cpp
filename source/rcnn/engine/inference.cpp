#include "inference.h"

#include <set>
#include <string>
#include <iostream>

#include <data.h>
#include <defaults.h>


namespace rcnn{
namespace engine{

using namespace rcnn::data;
using namespace rcnn::config;

void inference(){
  //Build Model
  GeneralizedRCNN model = BuildDetectionModel();
  
  //Build Dataset
  vector<string> dataset_list = GetCFG<std::vector<std::string>>({"DATASETS", "TEST"});
  Compose transforms = BuildTransforms(false);
  BatchCollator collate = BatchCollator(GetCFG<int>({"DATALOADER", "SIZE_DIVISIBILITY"}));
  COCODataset coco = BuildDataset(dataset_list, false);
  auto data = coco.map(transforms).map(collate);
  shared_ptr<torch::data::samplers::Sampler<>> sampler = make_batch_data_sampler(coco, false, 0);
  
  int images_per_batch = GetCFG<int64_t>({"TEST", "IMS_PER_BATCH"});
  torch::data::DataLoaderOptions options(images_per_batch);
  options.workers(GetCFG<int64_t>({"DATALOADER", "NUM_WORKERS"}));
  auto data_loader = torch::data::make_data_loader(std::move(data), *dynamic_cast<GroupedBatchSampler*>(sampler.get()), options);

  //Check iou type
  set<string> iou_types{"bbox"};
  if(GetCFG<bool>({"MODEL", "MASK_ON"}))
    iou_types.insert("segm");

  string output_folder = GetCFG<string>({"OUTPUT_DIR"});
  
  torch::Device device(GetCFG<string>({"MODEL", "DEVICE"}));

  //todo
  cout << "Start evaluation on dataset\n";
  Timer total_time = Timer();
  Timer inference_timer = Timer();

  total_time.tic();
  map<int64_t, BoxList> predictions = compute_on_dataset(model, data_loader, device, inference_timer);

  auto total_time_ = total_time.toc();
  string total_time_str = total_time.avg_time_str();

  cout << "Total run time: " << total_time_str << " (" << total_time_ * /*device num*/1 / coco.size().value() << " s / img per device, on 1 devices)\n";

  cout << "Model inference time: " << inference_timer.total_time.count() << "s (" << inference_timer.total_time.count() / coco.size().value() << " s / img per device, on 1 devices)\n";
  DoCOCOEvaluation(coco, predictions, output_folder, iou_types);
}

}
}