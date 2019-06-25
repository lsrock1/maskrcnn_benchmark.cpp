#include "trainer.h"
#include "defaults.h"
#include <modeling.h>

#include "transforms/build.h"
#include "samplers/build.h"
#include <bounding_box.h>
#include <build.h>
#include <tovec.h>
#include <collate_batch.h>

#include <solver_build.h>
#include <torch/torch.h>

#include <ctime>


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
  GeneralizedRCNN model = BuildDetectionModel();
  int max_iter = GetCFG<int64_t>({"SOLVER", "MAX_ITER"});
  int start_iter = iteration;
  model->train();
  time_t start_training_time = time(0);
  time_t end = time(0);
  double data_time;

  ConcatOptimizer optimizer = MakeOptimizer(model);
  ConcatScheduler scheduler = MakeLRScheduler(optimizer);
  
  vector<string> dataset_list;
  auto dataset = GetCFG<CFGS>({"DATASETS", "TRAIN"});
  dataset_list = tovec(dataset.get());
  Compose transforms = BuildTransforms(true);
  BatchCollator collate = BatchCollator(GetCFG<int>({"DATALOADER", "SIZE_DIVISIBILITY"}));
  int images_per_batch = GetCFG<int64_t>({"SOLVER", "IMS_PER_BATCH"});
  COCODataset coco = BuildDataset(dataset_list, true);
  auto data = coco.map(transforms).map(collate);
  std::shared_ptr<torch::data::samplers::Sampler<>> sampler = make_batch_data_sampler(coco, true, start_iter);


  torch::data::DataLoaderOptions options(images_per_batch);
  options.workers(GetCFG<int64_t>({"DATALOADER", "NUM_WORKERS"}));
  int num_iter = GetCFG<int64_t>({"SOLVER", "MAX_ITER"});
  auto data_loader = torch::data::make_data_loader(std::move(data), *dynamic_cast<IterationBasedBatchSampler*>(sampler.get()), options);
  for(auto& i : *data_loader){
    time(&end);
    data_time = difftime(time(0), end);
    iteration += 1;
    
    scheduler.step();
    ImageList images = get<0>(i).to(device);
    vector<BoxList> targets = get<1>(i);
    // for(auto& i : )
    //   targets.push_back(i.target);

    std::vector<rcnn::structures::BoxList> loss_map = model->forward<std::vector<rcnn::structures::BoxList>>(images, targets);

    // std::vector<torch::Tensor> losses;
    // for(auto i = loss_map.begin(); i != loss_map.end(); ++i)
    //   losses.push_back(i->second);
    // torch::Tensor loss = torch::zeros({1});
    // for(auto& i : losses)
    //   loss = loss + i;
    // optimizer.zero_grad();
    // loss.backward();
    // optimizer.step();
    // cout << loss.item<double>() << "\n";
  }
  

}

}
}