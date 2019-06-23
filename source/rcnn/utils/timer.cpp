#include "timer.h"
#include <sstream>


namespace rcnn{
namespace utils{

Timer::Timer() :start_time(time(0)), total_time(time(0)), diff(0.0), calls(0){}

void Timer::tic(){
  time(&start_time);
}

double Timer::toc(bool average){
  add(difftime(time(0), start_time));
  if(average){
    return average_time();
  }
  else{
    return diff;
  }
}

void Timer::add(double time_diff){
  diff = time_diff;
  total_time += diff;
  calls += 1;
}

void Timer::reset(){
  diff = 0;
  total_time = 0.0;
  calls = 0;
  start_time = 0.0;
}

double Timer::average_time(){
  return calls > 0 ? total_time / calls : 0.0;
}

std::string Timer::avg_time_str(){
  std::ostringstream strs;
  strs << average_time();
  return strs.str();
}

}
}