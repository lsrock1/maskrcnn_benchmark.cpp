#pragma once
#include <ctime>
#include <string>
#include <chrono>

namespace rcnn {
namespace utils {

class Timer {

public:
  Timer();
  void tic();
  double toc(bool average=true);
  void add(std::chrono::duration<double> time_diff);
  std::string avg_time_str();
  void reset();
  double average_time();
  std::chrono::duration<double> total_time;

private:
  std::chrono::system_clock::time_point start_time;
  double diff;
  int calls;
};

} // namespace utils
} // namespace rcnn
