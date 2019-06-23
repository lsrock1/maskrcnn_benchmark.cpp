#pragma once
#include <ctime>
#include <string>


namespace rcnn{
namespace utils{

class Timer{

public:
  Timer();
  void tic();
  double toc(bool average=true);
  void add(double time_diff);
  std::string avg_time_str();
  void reset();
  double average_time();

private:
  time_t start_time;
  time_t total_time;
  double diff;
  int calls;
};

}
}