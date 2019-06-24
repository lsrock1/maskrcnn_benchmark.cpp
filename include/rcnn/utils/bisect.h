#pragma once
#include <vector>
#include <cstdint>


namespace rcnn{
namespace utils{

double bisect_right(std::vector<int64_t> milestones, int64_t element);
double bisect_right(std::vector<float> milestones, float element);


}
}