#include "gtest/gtest.h"

#include <defaults.h>

using namespace rcnn::config;

TEST(defaults, SetCFGFromFile)
{
  //check if default empty, yaml exists
  CFGS model_weight = GetCFG<CFGS>({"MODEL", "WEIGHT"});
  EXPECT_STREQ("catalog://ImageNetPretrained/MSRA/R-50", model_weight.get());

  //check if default exists, yaml exists
  float lr = GetCFG<float>({"SOLVER", "BASE_LR"});
  EXPECT_EQ(lr, static_cast<float>(0.01));
    
  //check if default exists, yaml empty
  bool stride1x1 = GetCFG<bool>({"MODEL", "RESNETS", "STRIDE_IN_1X1"});
  EXPECT_EQ(stride1x1, true);

  //check vector<int64_t>
  std::vector<int64_t> int64t_vec = GetCFG<std::vector<int64_t>>({"MODEL", "ROI_MASK_HEAD", "CONV_LAYERS"});
  for(auto& i : int64t_vec)
    EXPECT_EQ(i, 256);

  //check vector<float>
  std::vector<float> float_vec = GetCFG<std::vector<float>>({"MODEL", "ROI_HEADS", "BBOX_REG_WEIGHTS"});
  EXPECT_EQ(float_vec[0], 10.);
  EXPECT_EQ(float_vec[1], 10.0);
  EXPECT_EQ(float_vec[2], 5.0);
  EXPECT_EQ(float_vec[3], 5.0);
}

// TEST(defaults, SetCFGFromFile)
// {
//   //check if default empty, yaml exists
//   CFGS model_weight = GetCFG<CFGS>({"MODEL", "WEIGHT"});
//   EXPECT_STREQ("catalog://ImageNetPretrained/MSRA/R-50", model_weight.get());

//   //check if default exists, yaml exists
//   float lr = GetCFG<float>({"SOLVER", "BASE_LR"});
//   EXPECT_EQ(lr, static_cast<float>(0.01));
    
//   //check if default exists, yaml empty
//   bool stride1x1 = GetCFG<bool>({"MODEL", "RESNETS", "STRIDE_IN_1X1"});
//   EXPECT_EQ(stride1x1, true);
// }