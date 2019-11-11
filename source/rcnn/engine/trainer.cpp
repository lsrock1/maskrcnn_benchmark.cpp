#include "trainer.h"
#include "parallel.h"
#include "defaults.h"
#include <modeling.h>

#include "transforms/build.h"
#include "samplers/build.h"
#include <bounding_box.h>
#include <build.h>
#include <metric_logger.h>
#include <checkpoint.h>
#include <collate_batch.h>

#include <solver_build.h>
#include <torch/torch.h>

#include <ctime>
#include <chrono>
#include <iostream>

using namespace rcnn::modeling;
using namespace rcnn::config;
using namespace rcnn::data;
using namespace rcnn::solver;
using namespace rcnn::structures;
using namespace rcnn::utils;
using namespace std;

//TODO logger
namespace rcnn {
namespace engine {

void do_train() {
  //meters
  auto meters = MetricLogger(" ");
  torch::Device device(GetCFG<std::string>({"MODEL", "DEVICE"}));

  string output_dir = GetCFG<std::string>({"OUTPUT_DIR"});
  int max_iter = GetCFG<int64_t>({"SOLVER", "MAX_ITER"});
  auto start_training_time = chrono::system_clock::now();
  auto end = chrono::system_clock::now();
  chrono::duration<double> data_time, batch_time;
  float eta_seconds;
  string eta_string;
  int days, hours, minutes;
  int checkpoint_period = GetCFG<int>({"SOLVER", "CHECKPOINT_PERIOD"});
  cout << "building model\n";
  GeneralizedRCNN model = BuildDetectionModel();
  cout << "build complete!\n";

  cout << "Making optimizer and scheduler\n";
  ConcatOptimizer optimizer = MakeOptimizer(model);
  ConcatScheduler scheduler = MakeLRScheduler(optimizer, 0);
  auto check_point = Checkpoint(model, optimizer, scheduler, output_dir);
  int start_iter = check_point.load(GetCFG<std::string>({"MODEL", "WEIGHT"}));
  int iteration = start_iter;
  scheduler.set_last_epoch(start_iter);
  vector<string> dataset_list = GetCFG<std::vector<std::string>>({"DATASETS", "TRAIN"});
  Compose transforms = BuildTransforms(true);
  BatchCollator collate = BatchCollator(GetCFG<int>({"DATALOADER", "SIZE_DIVISIBILITY"}));
  int images_per_batch = GetCFG<int64_t>({"SOLVER", "IMS_PER_BATCH"});
  COCODataset coco = BuildDataset(dataset_list, true);

  auto data = coco.map(transforms).map(collate);
  shared_ptr<torch::data::samplers::Sampler<>> sampler = make_batch_data_sampler(coco, true, start_iter);
  
  torch::data::DataLoaderOptions options(images_per_batch);
  options.workers(GetCFG<int64_t>({"DATALOADER", "NUM_WORKERS"}));
  auto data_loader = torch::data::make_data_loader(std::move(data), *dynamic_cast<IterationBasedBatchSampler*>(sampler.get()), options);
  
  model->to(device);
  model->train();
  cout << "Start training\n";
  for (auto& i : *data_loader) {
    data_time = chrono::system_clock::now() - end;
    iteration += 1;
    scheduler.step();
    
    #ifdef WITH_CUDA
    map<string, torch::Tensor> loss_map;
    torch::Tensor loss;
    std::tie(loss, loss_map) = data_parallel(model, get<0>(i), get<1>(i));

    #else
    ImageList images = get<0>(i).to(device);
    vector<BoxList> targets;
    
    for (auto& target : get<1>(i)) {
      targets.push_back(target.To(device));
    }

    map<string, torch::Tensor> loss_map = model->forward<map<string, torch::Tensor>>(images, targets);

    for (auto i = loss_map.begin(); i != loss_map.end(); ++i) {
      if (i == loss_map.begin()) {
        loss_map["loss"] = i->second;
      }
      else {
        loss_map["loss"] = loss_map["loss"] + i->second;
      }
    }
    torch::Tensor loss = loss_map["loss"];
    
    #endif    
    meters.update(loss_map);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    batch_time = chrono::system_clock::now() - end;
    end = chrono::system_clock::now();
    meters.update(map<string, float>{{"time", static_cast<float>(batch_time.count())}, {"data", static_cast<float>(data_time.count())}});
    eta_seconds = meters["time"].global_avg() * (max_iter - iteration);
    days = eta_seconds/60/60/24;
    hours = eta_seconds/60/60 - days* 24;
    minutes = eta_seconds/60 - hours * 60 - days * 24 * 60;
    eta_string = to_string(days) + " day " + to_string(hours) + " h " + to_string(minutes) + " m";
    if (iteration % 20 == 0 || iteration == max_iter) {
      cout << "eta: " << eta_string << meters.delimiter_ << "iter: " << iteration << meters.delimiter_ << meters << meters.delimiter_ << "lr: " << to_string(optimizer.get_lr()) << meters.delimiter_ << "max mem: " << "none\n";
    }

    if (iteration % checkpoint_period == 0) {
      check_point.save("model_" + to_string(iteration) + ".pth", iteration);
    }

    if (iteration == max_iter) {
      check_point.save("model_final.pth", iteration);
    }
  }
  
  chrono::duration<double> total_training_time = chrono::system_clock::now() - start_training_time;
  days = total_training_time.count()/60/60/24;
  hours = total_training_time.count()/60/60 - days* 24;
  minutes = total_training_time.count()/60 - hours * 60 - days * 24 * 60;
  cout << "Total training time: " << to_string(days) + " day " + to_string(hours) + " h " + to_string(minutes) + " m" << " ( " << total_training_time.count() / max_iter << "s / it)";
}

} // namespace engine
} // namespace rcnn
