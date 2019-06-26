#include "trainer.h"
#include "defaults.h"
#include <modeling.h>

#include "transforms/build.h"
#include "samplers/build.h"
#include <bounding_box.h>
#include <build.h>
#include <metric_logger.h>
#include <collate_batch.h>

#include <solver_build.h>
#include <torch/torch.h>

#include <ctime>
#include <iostream>


//TODO logger
namespace rcnn{
namespace engine{

using namespace rcnn::modeling;
using namespace rcnn::config;
using namespace rcnn::data;
using namespace rcnn::solver;
using namespace rcnn::structures;
// using namespace rcnn::utils;
using namespace std;

void do_train(int checkpoint_period, int iteration, torch::Device device){
  //meters
  auto meters = MetricLogger(" ");
  cout << "Start training\n";
  int max_iter = GetCFG<int64_t>({"SOLVER", "MAX_ITER"});
  int start_iter = iteration;
  time_t start_training_time = time(0);
  time_t end = time(0);
  double data_time, batch_time, eta_seconds;
  std::string eta_string;

  GeneralizedRCNN model = BuildDetectionModel();
  model->to(device);
  model->train();
  ConcatOptimizer optimizer = MakeOptimizer(model);
  ConcatScheduler scheduler = MakeLRScheduler(optimizer, start_iter);
  
  vector<string> dataset_list = GetCFG<std::vector<std::string>>({"DATASETS", "TRAIN"});
  Compose transforms = BuildTransforms(true);
  BatchCollator collate = BatchCollator(GetCFG<int>({"DATALOADER", "SIZE_DIVISIBILITY"}));
  int images_per_batch = GetCFG<int64_t>({"SOLVER", "IMS_PER_BATCH"});
  COCODataset coco = BuildDataset(dataset_list, true);

  auto data = coco.map(transforms).map(collate);
  std::shared_ptr<torch::data::samplers::Sampler<>> sampler = make_batch_data_sampler(coco, true, start_iter);
  
  torch::data::DataLoaderOptions options(images_per_batch);
  options.workers(GetCFG<int64_t>({"DATALOADER", "NUM_WORKERS"}));
  auto data_loader = torch::data::make_data_loader(std::move(data), *dynamic_cast<IterationBasedBatchSampler*>(sampler.get()), options);
  
  for(auto& i : *data_loader){
    time(&end);
    data_time = difftime(time(0), end);
    iteration += 1;
    scheduler.step();
    ImageList images = get<0>(i).to(device);
    vector<BoxList> targets;
    
    for(auto& target : get<1>(i))
      targets.push_back(target.To(device));

    cout << images.get_tensors().sizes();
    map<string, torch::Tensor> loss_map = model->forward<map<string, torch::Tensor>>(images, targets);
    torch::Tensor loss = torch::zeros({1}).to(device);

    for(auto i = loss_map.begin(); i != loss_map.end(); ++i)
      loss += i->second;
    loss_map["loss"] = loss;
    meters.update(loss_map);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    batch_time = difftime(time(0), end);
    end = time(0);
    meters.update(map<string, float>{{"time", static_cast<float>(batch_time)}, {"data", static_cast<float>(data_time)}});
    eta_second = meters["time"].global_avg() * (max_iter - iteration);
    eta_string = std::to_string(eta_seconds/60/60/24) + " day " + std::to_string(eta_second/60/60) + " h " + std::string(eta_second/60) + " m";
    if(iteration % 20 == 0 || iteration == max_iter){
      cout << "eta: " << eta_string << meters.delimiter_ << "iter: " << iteration << meters.delimiter_ << meters << meters.delimiter_
      << "lr: " << to_string(optimizer.get_lr()) << meters.delimiter_ << "max mem: " << "none\n";
    }
  }
  

}

}
}