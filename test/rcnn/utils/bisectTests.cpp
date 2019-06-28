#include "gtest/gtest.h"

#include <bisect.h>

using namespace rcnn::utils;

TEST(utils, bisect)
{
  ASSERT_EQ(bisect_right(std::vector<int64_t> {3, 6, 9}, 5), 1);
  ASSERT_EQ(bisect_right(std::vector<int64_t> {3, 6, 9}, 3), 1);
  ASSERT_EQ(bisect_right(std::vector<int64_t> {3, 6, 9}, 12), 3);
}