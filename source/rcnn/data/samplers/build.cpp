#include "samplers/build.h"
#include <memory>
#include "bisect.h"
#include "samplers/samplers.h"

#include <defaults.h>


namespace rcnn{
namespace data{

std::shared_ptr<torch::data::samplers::Sampler<>> make_data_sampler(int dataset_size, bool shuffle/*, distributed*/){
  if(shuffle)
    return std::shared_ptr<torch::data::samplers::Sampler<>> (new torch::data::samplers::RandomSampler(dataset_size));
  else
    return std::shared_ptr<torch::data::samplers::Sampler<>> (new torch::data::samplers::SequentialSampler(dataset_size));
}

std::vector<int> _quantize(std::vector<float> x, std::vector<float> bins){
  std::vector<int> quantized;
  for(auto& i : x)
    quantized.push_back(static_cast<int>(rcnn::utils::bisect_right(bins, i)));
}

std::vector<float> _compute_aspect_ratios(COCODataset dataset){
  std::vector<float> aspect_ratios;
  float aspect_ratio;
  for(int i = 0; i < *dataset.size(); ++i){
    auto image_info = dataset.get_img_info(i);
    aspect_ratio = static_cast<float>(image_info.height) / static_cast<float>(image_info.width);
    aspect_ratios.push_back(aspect_ratio);
  }
  return aspect_ratios;
}

std::shared_ptr<torch::data::samplers::Sampler<>> make_batch_data_sampler(COCODataset dataset, 
                                                                          bool is_train,
                                                                          int start_iter)
{
  std::shared_ptr<torch::data::samplers::Sampler<>> batch_sampler;
  int64_t images_per_batch;
  bool shuffle = true;
  int num_iters = -1;
  if(is_train){
    images_per_batch = rcnn::config::GetCFG<int64_t>({"SOLVER", "IMS_PER_BATCH"});
    // shuffle = true;
    num_iters = rcnn::config::GetCFG<int64_t>({"SOLVER", "MAX_ITER"});
  }
  else{
    images_per_batch = rcnn::config::GetCFG<int64_t>({"TEST", "IMS_PER_BATCH"});
    //no distributed
    start_iter = 0;
  }
  bool aspect_grouping = rcnn::config::GetCFG<bool>({"DATALOADER", "ASPECT_RATIO_GROUPING"});

  std::shared_ptr<torch::data::samplers::Sampler<>> sampler = make_data_sampler(dataset.size().value(), shuffle);
  if(aspect_grouping){
    std::vector<float> aspect_ratios = _compute_aspect_ratios(dataset);
    std::vector<int> group_ids = _quantize(aspect_ratios, std::vector<float>{1});
    batch_sampler = std::make_shared<GroupedBatchSampler>(sampler, group_ids, images_per_batch, false);
  }
  else{
    batch_sampler = sampler;
  }
  
  if(num_iters != -1){
    batch_sampler = std::make_shared<IterationBasedBatchSampler>(batch_sampler, num_iters, start_iter);
  }

  return batch_sampler;
}
                                                                        

}
}
