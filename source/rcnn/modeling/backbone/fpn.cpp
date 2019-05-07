#include "fpn.h"

namespace rcnn{
namespace modeling{
  class FPNImpl : public torch::nn::Module{
    private:
      bool use_relu;
      int64_t out_channels;
      std::string last_level;
      torch::nn::Conv2d inner_block1_, inner_block2_, inner_block3_, inner_block4{nullptr};
      torch::nn::Conv2d layer_block1_, layer_block2_, layer_block3_, layer_block4{nullptr};

    public:
      FPNImpl(int64_t in_channels_list, int64_t out_channels, std::string last_level=nullptr);
      torch::Tensor forward(std::vector<torch::Tensor> x);
  }

  TORCH_MODULE(FPN);
}
}