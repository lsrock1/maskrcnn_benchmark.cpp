#include "gtest/gtest.h"

#include <torch/torch.h> 
#include <modeling.h>
#include <defaults.h>


using namespace rcnn::modeling;
using namespace rcnn::config;

TEST(backbone, build)
{
  std::vector<std::string> cfg_files{"../resource/e2e_faster_rcnn_R_50_C4_1x.yaml", "../resource/e2e_faster_rcnn_R_50_FPN_1x.yaml"};
  
  for(auto& file : cfg_files){
    SetCFGFromFile(file);
    auto backbone = BuildBackbone();
    EXPECT_GT(backbone->get_out_channels(), 0);

    int64_t size = 256;
    auto input = torch::zeros({2, 3, size, size}).to(torch::kF32);
    auto out = backbone->forward(input);
    
    //check out feature map num is equal with config file
    auto strides = GetCFG<std::vector<int64_t>>({"MODEL", "RPN", "ANCHOR_STRIDE"});
    ASSERT_EQ(out.size(), strides.size());

    for(int i = 0; i < out.size(); ++i){
      ASSERT_EQ(out[i].size(0), 2);
      ASSERT_EQ(out[i].size(1), backbone->get_out_channels());
      ASSERT_EQ(out[i].size(2), size / strides[i]);
      ASSERT_EQ(out[i].size(3), size / strides[i]);
    }
  }

}