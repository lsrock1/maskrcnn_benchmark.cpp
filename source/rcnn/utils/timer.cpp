#include "timer.h"
#include <sstream>


namespace rcnn{
namespace utils{

Timer::Timer() :diff(0.0), calls(0){
  start_time = std::chrono::system_clock::now();
  total_time = std::chrono::duration<double>::zero();
}

void Timer::tic(){
  start_time = std::chrono::system_clock::now();
}

double Timer::toc(bool average){
  add(std::chrono::system_clock::now() - start_time);
  if(average){
    return average_time();
  }
  else{
    return diff;
  }
}

void Timer::add(std::chrono::duration<double> time_diff){
  diff = time_diff.count();
  total_time += time_diff;
  calls += 1;
}

void Timer::reset(){
  diff = 0;
  total_time = std::chrono::duration<double>::zero();
  calls = 0;
  start_time = std::chrono::system_clock::now();
}

double Timer::average_time(){
  return calls > 0 ? total_time.count() / calls : 0.0;
}

std::string Timer::avg_time_str(){
  std::ostringstream strs;
  strs << average_time();
  return strs.str();
}

}
}