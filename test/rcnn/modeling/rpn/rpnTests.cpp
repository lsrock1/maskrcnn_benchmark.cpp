#include "gtest/gtest.h"

#include <torch/torch.h> 
#include <modeling.h>
#include <defaults.h>


using namespace rcnn::modeling;
using namespace rcnn::config;

TEST(rpn, head)
{
  //rpnmodule is only one
  int64_t in_channels = 64;
  int64_t num_anchors = 10;

  auto rpn = RPNHead(in_channels, num_anchors);

  std::vector<torch::Tensor> input{torch::zeros({2, in_channels, 24, 32}).to(torch::kF32),
                                   torch::zeros({2, in_channels, 24, 32}).to(torch::kF32),
                                   torch::zeros({2, in_channels, 24, 32}).to(torch::kF32)};

  auto output = rpn->forward(input);
  std::vector<torch::Tensor> logits;
  std::vector<torch::Tensor> regression;
  std::tie(logits, regression) = output;

  for(int i = 0; i < 3; ++i){
    ASSERT_EQ(logits[i].size(0), input[i].size(0));
    ASSERT_EQ(logits[i].size(1), num_anchors);
    ASSERT_EQ(logits[i].size(2), input[i].size(2));
    ASSERT_EQ(logits[i].size(3), input[i].size(3));

    ASSERT_EQ(regression[i].size(0), input[i].size(0));
    ASSERT_EQ(regression[i].size(1), num_anchors*4);
    ASSERT_EQ(regression[i].size(2), input[i].size(2));
    ASSERT_EQ(regression[i].size(3), input[i].size(3));
  }
}

TEST(rpn, anchor)
{
  auto anchor_generator = MakeAnchorGenerator();
  int64_t image_width = 256;
  int64_t image_num = 2;
  std::vector<torch::Tensor> feature_maps;
  std::vector<std::pair<int64_t, int64_t>> grid_sizes;

  rcnn::structures::ImageList imageList(torch::zeros({1}), std::vector<std::pair<int64_t, int64_t>>{std::make_pair(256, 256), std::make_pair(256, 256)});
  
  //anchor_generator(imageList, )
  auto strides = GetCFG<std::vector<int64_t>>({"MODEL", "RPN", "ANCHOR_STRIDE"});
  for(auto& stride : strides){
    feature_maps.push_back(torch::zeros({image_num, 1, image_width/stride, image_width/stride}));
    grid_sizes.push_back(std::make_pair(image_width/stride, image_width/stride));
  }

  auto generated_anchors = anchor_generator->forward(imageList, feature_maps);

  ASSERT_EQ(generated_anchors.size(), image_num);
  ASSERT_EQ(generated_anchors[0].size(), strides.size());
  for(auto& image : generated_anchors){
    for(int i = 0; i < image.size(); ++i){
      int64_t mul = GetCFG<std::vector<float>>({"MODEL", "RPN", "ASPECT_RATIOS"}).size();
      if(GetCFG<std::vector<int64_t>>({"MODEL", "RPN", "ANCHOR_STRIDE"}).size() == 1){
        auto sizes = GetCFG<std::vector<int64_t>>({"MODEL", "RPN", "ANCHOR_SIZES"});
        mul *= sizes.size();
      }
      ASSERT_EQ(image[i].get_bbox().size(0), std::get<0>(grid_sizes[i]) * std::get<1>(grid_sizes[i]) * mul);
    }
  }
}

TEST(rpn, anchor_generate)
{
  // int64_t stride = 16;
  // vector<int64_t> anchor_sizes{8, 16, 32};
  // vector<float> aspect_ratios{0.5, 1, 2};
  torch::Tensor anchors = GenerateAnchors(16, std::vector<int64_t> {8, 16, 32}, std::vector<float> {0.5, 1, 2});
  torch::Tensor corrects = torch::tensor({
    2.2500,   5.0000,  12.7500,  10.0000,
    -3.5000,   2.0000,  18.5000,  13.0000,
    -15.0000,  -4.0000,  30.0000,  19.0000,
    4.0000,   4.0000,  11.0000,  11.0000,
    0.0000,   0.0000,  15.0000,  15.0000,
    -8.0000,  -8.0000,  23.0000,  23.0000,
    5.2500,   2.5000,   9.7500,  12.5000,
    2.5000,  -3.0000,  12.5000,  18.0000,
    -3.0000, -14.0000,  18.0000,  29.0000
  }).to(torch::kF32);
  anchors = anchors.reshape({-1});
  for(int i = 0; i < anchors.size(0); ++i){
    ASSERT_EQ(anchors[i].item<float>(), corrects[i].item<float>());
  }
}