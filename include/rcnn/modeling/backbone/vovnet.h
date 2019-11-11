#pragma once
#include <conv2d.h>
#include <batch_norm.h>

namespace rcnn {
namespace modeling {

class OSA_moduleImpl : public torch::nn::Module {

public:
  OSA_moduleImpl(int64_t in_channels, int64_t stage_channels, int64_t concat_channels, int layers_per_block, std::string module_name, bool identity = false);
  torch::Tensor forward(torch::Tensor x);

private:
  bool identity_;
  std::vector<torch::nn::Sequential> layers_;
  torch::nn::Sequential concat_;
};

TORCH_MODULE(OSA_module);

class OSA_stageImpl : public torch::nn::Module {

public:
  OSA_stageImpl(int64_t in_channels, int64_t stage_channels, int64_t concat_channels, int blocks_per_stage, int layers_per_block, int stage_num);
  torch::Tensor forward(torch::Tensor x);

private:
  torch::nn::Sequential modules;

};

TORCH_MODULE(OSA_stage);

class VoVNetImpl : public torch::nn::Module {

public:
  struct StageSpec{
    StageSpec(std::vector<int> stage_channels, std::vector<int> concat_channels, int layers_per_block, std::vector<int> blocks_per_stage);
    std::vector<int> stage_channels_;
    std::vector<int> concat_channels_;
    int layers_per_block_;
    std::vector<int> blocks_per_stage_;
  };

  VoVNetImpl();
  std::vector<torch::Tensor> forward(torch::Tensor x);
  std::shared_ptr<torch::nn::Module> clone(const torch::optional<torch::Device>& device = torch::nullopt) const override;

private:
  void initialize_weights();
  void freeze_backbone(int freeze_at);
  std::vector<OSA_stage> stages_;
  torch::nn::Sequential stem_;
};

TORCH_MODULE(VoVNet);

void conv3x3(torch::nn::Sequential& seq,
             int64_t in_channels,
             int64_t out_channels,
             const std::string& module_name,
             const std::string& postfix,
             int64_t stride = 1,
             int64_t groups = 1,
             int64_t kernel_size = 3,
             int64_t padding = 1);

void conv1x1(torch::nn::Sequential& seq,
             int64_t in_channels,
             int64_t out_channels,
             const std::string& module_name,
             const std::string& postfix,
             int64_t stride = 1,
             int64_t groups = 1,
             int64_t kernel_size = 1,
             int64_t padding = 0);

} // namespace modeling

namespace registry{

// const _STEM_MODULES
rcnn::modeling::VoVNetImpl::StageSpec STAGE_SPECS_VoVNet(std::string name);

} // namespace registry
} // namespace rcnn
