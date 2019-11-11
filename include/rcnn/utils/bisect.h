#pragma once
#include <vector>
#include <cstdint>

namespace rcnn {
namespace utils {

/// not actually bisect
inline double bisect_right(std::vector<int64_t> milestones, int64_t element) {
  int index = 0;
  for (auto& i : milestones) {
    if (element < i)
      return index;
    else if (element == i)
      return index + 1;
    else
      index += 1;
  }
  return static_cast<double>(index);
}

inline double bisect_right(std::vector<float> milestones, float element) {
  int index = 0;
  for (auto& i : milestones) {
    if (element < i)
      return index;
    else if (element == i)
      return index + 1;
    else
      index += 1;
  }
  return static_cast<double>(index);
}

} // namespace utils
} // namespace rcnn
