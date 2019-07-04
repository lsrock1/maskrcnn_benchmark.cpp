#pragma once
#include <map>
#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <bounding_box.h>
#include <modeling.h>
#include <timer.h>

#include <torch/torch.h>


namespace rcnn{
namespace engine{

using namespace rcnn::structures;
using namespace rcnn::modeling;
using namespace std;
using namespace rcnn::utils;

template<typename Dataset>
map<int64_t, BoxList> compute_on_dataset(GeneralizedRCNN& model, Dataset& dataset, torch::Device& device, Timer& inference_timer, int total_size){
  torch::NoGradGuard guard;
  model->eval();
  model->to(device);
  cout << "Model to Device\n";
  map<int64_t, BoxList> results_map;
  torch::Device cpu_device = torch::Device("cpu");
  int progress = 0;
  for(auto& batch : *dataset){
    vector<BoxList> output;
    ImageList images = get<0>(batch).to(device);
    vector<int64_t> image_ids = get<2>(batch);
    inference_timer.tic();
    output = model->forward(images);
    inference_timer.toc();
    for(auto& i : output)
      i = i.To(cpu_device);
    assert(output.size() == image_ids.size());
    for(int i = 0; i < output.size(); ++i)
      results_map.insert({image_ids[i], output[i]});
    progress += images.get_tensors().size(0);
    std::cout << progress << "/" << total_size << "\n";
  }
  return results_map;
}

void inference();

}
}