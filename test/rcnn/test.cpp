#include "gtest/gtest.h"
#include "defaults.h"


int main(int argc, char* argv[])
{
    rcnn::config::SetCFGFromFile("../e2e_faster_rcnn_R_50_C4_1x.yaml");
    testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    return ret;
}