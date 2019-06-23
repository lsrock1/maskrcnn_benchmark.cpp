#include "samplers/samplers.h"


namespace rcnn{
namespace data{

GroupedBatchSampler::GroupedBatchSampler(std::shared_ptr<torch::data::samplers::Sampler<>> sampler, 
                                         std::vector<int> group_ids, 
                                         int batch_size, 
                                         bool drop_uneven)
                                        :sampler_(sampler),
                                         group_ids_(init_group_id(group_ids)),
                                         batch_size_(batch_size),
                                         drop_uneven_(drop_uneven),
                                         groups(std::get<0>(std::get<0>(torch::_unique(group_ids_)).sort(0))),
                                         _can_reuse_batches(false){}

torch::Tensor GroupedBatchSampler::init_group_id(std::vector<int> group_ids){
  torch::Tensor group_tensor = torch::zeros({static_cast<int64_t>(group_ids.size())});
  for(int i = 0; i < group_ids.size(); ++i)
    group_tensor[i] = group_ids[i];

  return group_tensor;
}

std::vector<torch::Tensor> GroupedBatchSampler::_prepare_batches(){
  int64_t dataset_size = group_ids_.size(0);
  std::vector<int> sampled_ids;
  auto i = sampler_->next(1);
  while(i.has_value()){
    sampled_ids.push_back(i.value()[0]);
    i = sampler_->next(1);
  }
  sampled_ids_ = torch::zeros({static_cast<int64_t>(sampled_ids.size())});
  for(int i = 0; i < sampled_ids.size(); ++i)
    sampled_ids_[i] = sampled_ids[i];

  torch::Tensor order = torch::full({dataset_size}, -1).to(torch::kI64);
  order.index_copy_(0, sampled_ids_, torch::arange(sampled_ids_.size(0)));

  torch::Tensor mask = order >= 0;
  std::vector<torch::Tensor> clusters;
  clusters.reserve(groups.size(0));
  for(int i = 0; i < groups.size(0); ++i)
    clusters.push_back((group_ids_ == groups[i]).__and__(mask));

  //clusters.clear();
  std::vector<torch::Tensor> relative_order;
  relative_order.reserve(clusters.size());
  for(auto& cluster: clusters)
    clusters.push_back(order.index_select(0, cluster));

  std::vector<torch::Tensor> permutation_ids;
  permutation_ids.reserve(relative_order.size());
  for(auto& s : relative_order)
    permutation_ids.push_back(s.index_select(0, std::get<1>(s.sort())));

  std::vector<torch::Tensor> permuted_clusters;
  permuted_clusters.reserve(permutation_ids.size());
  for(auto& idx : permutation_ids)
    permuted_clusters.push_back(sampled_ids_.index_select(0, idx));

  std::vector<torch::Tensor> merged;
  merged.reserve(permuted_clusters.size());
  for(auto& c : permuted_clusters){
    auto splitted_c = c.split(batch_size_);
    merged.reserve(merged.size() + splitted_c.size());
    merged.insert(merged.end(), splitted_c.begin(), splitted_c.end());
  }

  std::vector<int> first_element_of_batch;
  first_element_of_batch.reserve(merged.size());
  for(auto& t : merged)
    first_element_of_batch.push_back(t[0].item<int>());
    
  std::map<int, int> inv_sampled_ids_map;
  for(int i = 0; i < sampled_ids_.size(0); ++i)
    inv_sampled_ids_map[sampled_ids_[i].item<int>()] = i;

  torch::Tensor first_index_of_batch = torch::zeros({static_cast<int64_t>(first_element_of_batch.size())}).to(torch::kI64);
  for(int i = 0; i < first_element_of_batch.size(); ++i)
    first_index_of_batch[i] = inv_sampled_ids_map[first_element_of_batch[i]];

  std::vector<int> permutation_order;
  auto sorted_indices = std::get<1>(first_index_of_batch.sort(0));
  permutation_order.reserve(sorted_indices.size(0));
  for(int i = 0; i < sorted_indices.size(0); ++i)
    permutation_order.push_back(sorted_indices[i].item<int>());

  std::vector<torch::Tensor> batches;
  batches.reserve(permutation_order.size());
  for(auto& i : permutation_order)
    batches.push_back(merged[i]);

  if(drop_uneven_){
    for(auto i = batches.begin(); i != batches.end();){
      if(i->size(0) == batch_size_)
        i++;
      else
        i = batches.erase(i);
    }
  }
  return batches;
}

torch::optional<std::vector<size_t>> GroupedBatchSampler::next(size_t batch_size){
  if(_batches.size() == 0)
    _batches = _prepare_batches();

  if(index_ <= _batches.size()){
    std::vector<size_t> indices;
    auto indices_tensor = _batches[index_];
    for(int i = 0; i < indices_tensor.size(0); ++i)
      indices.push_back(indices_tensor[i].item<int>());
    index_++;
    return indices;
  }

  return torch::nullopt;
}

void GroupedBatchSampler::reset(torch::optional<size_t> new_size){
  _batches = _prepare_batches();
  index_ = 0;
}

void GroupedBatchSampler::load(torch::serialize::InputArchive& archive){
  auto tensor = torch::empty(1, torch::kInt64);
  archive.read(
      "index",
      tensor,
      true);
  index_ = tensor.item<int64_t>();

  archive.read(
      "batch_size",
      tensor,
      true
  );
  batch_size_ = tensor.item<int64_t>();

  archive.read(
      "group_id",
      group_ids_,
      true
  );
  archive.read(
      "sample_id",
      sampled_ids_,
      true
  );
  archive.read(
      "groups",
      groups,
      true
  );
}

void GroupedBatchSampler::save(torch::serialize::OutputArchive& archive) const {
  archive.write(
      "index",
      torch::tensor(static_cast<int64_t>(index_), torch::kI64),
      true
  );
  archive.write(
      "batch_size",
      torch::tensor(static_cast<int64_t>(batch_size_), torch::kI64),
      true
  );
  archive.write(
      "group_id",
      group_ids_,
      true
  );
  archive.write(
      "sample_id",
      sampled_ids_,
      true
  );
  archive.write(
      "groups",
      groups,
      true
  );
}

}
}