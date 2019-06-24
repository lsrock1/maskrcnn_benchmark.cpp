#pragma once
#include <memory>

#include <torch/torch.h>
#include <torch/data/samplers/base.h>

namespace rcnn{
namespace data{

class GroupedBatchSampler : public torch::data::samplers::Sampler<>{

public:
  GroupedBatchSampler(std::shared_ptr<torch::data::samplers::Sampler<>> sampler, std::vector<int> group_ids, int batch_size, bool drop_uneven=false);
  void reset(torch::optional<size_t> new_size) override;
  torch::optional<std::vector<size_t>> next(size_t batch_size) override;
  void save(torch::serialize::OutputArchive& archive) const override;
  void load(torch::serialize::InputArchive& archive) override;

private:
  torch::Tensor init_group_id(std::vector<int> group_ids);
  std::vector<torch::Tensor> _prepare_batches();

  std::shared_ptr<torch::data::samplers::Sampler<>> sampler_;
  torch::Tensor group_ids_, sampled_ids_;
  int batch_size_;
  bool drop_uneven_;
  torch::Tensor groups;
  std::vector<torch::Tensor> _batches;
  bool _can_reuse_batches = false;
  int index_ = 0;
};

class IterationBasedBatchSampler : public torch::data::samplers::Sampler<>{

public:
  IterationBasedBatchSampler(std::shared_ptr<torch::data::samplers::Sampler<>> sampler, int num_iterations, int start_iter = 0);
  void reset(torch::optional<size_t> new_size) override;
  torch::optional<std::vector<size_t>> next(size_t batch_size) override;
  void save(torch::serialize::OutputArchive& archive) const override;
  void load(torch::serialize::InputArchive& archive) override;

private:
  std::shared_ptr<torch::data::samplers::Sampler<>> sampler_;
  int num_iterations_;
  int index_;
};

}
}