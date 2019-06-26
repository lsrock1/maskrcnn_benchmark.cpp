#include "metric_logger.h"


namespace rcnn{
namespace utils{

void SmoothedValue::update(float value){
  if(deque.size() < 21)
    deque.push_back(value);
  else
    deque.pop_front();
  count += 1;
  total += value;
}

float SmoothedValue::median() const{
  auto d = torch::zeros({static_cast<int64_t>(deque.size())});
  for(int i = 0; i < deque.size(); ++i)
    d[i] = deque[i];
  return d.median().item<float>();
}

float SmoothedValue::avg() const{
  auto d = torch::zeros({static_cast<int64_t>(deque.size())});
  for(int i = 0; i < deque.size(); ++i)
    d[i] = deque[i];
  return d.mean().item<float>();
}

float SmoothedValue::global_avg() const{
  return total / count;
}

MetricLogger::MetricLogger(std::string delimiter) : delimiter_(delimiter){}

void MetricLogger::update(std::map<std::string, torch::Tensor> losses){
  for(auto i = losses.begin(); i != losses.end(); ++i)
    meters[i->first].update(i->second.item<float>());
}

void MetricLogger::update(std::map<std::string, float> losses){
  for(auto i = losses.begin(); i != losses.end(); ++i)
    meters[i->first].update(i->second);
}

SmoothedValue MetricLogger::operator[](std::string attr){
  return meters[attr];
}

std::ostream& operator << (std::ostream& os, const MetricLogger& bml){
  for(auto i = bml.meters.begin(); i != bml.meters.end(); ++i){
    os << i->first << ": " << i->second.median() << " (" << i->second.global_avg() << ")";
    if(++i != bml.meters.end()){
      os << bml.delimiter_;
      i--;
    }
  }
  os << "\n";
  return os;
}

}
}