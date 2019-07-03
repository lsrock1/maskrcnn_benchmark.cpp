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
  int64_t total = 0;
  int64_t first = 0;
  int64_t second = 0;
  int64_t third = 0;
  // int index = 0;
  for(auto& batch : *dataset){
    vector<BoxList> output;
    // torch::Tensor tmp = get<0>(batch).get_tensors();
    // for(int i = 0; i < tmp.size(0); ++i){
    //   tmp.select(1, 0).fill_(255);
    //   cv::Mat warp(tmp.size(2), tmp.size(3), CV_32FC3, tmp[i].permute({1, 2, 0}).contiguous().data<float>());
    //   cv::imwrite("../resource/tmp/" + std::to_string(index) + std::to_string(i) + ".jpg", warp);
    // }
    // index++;
    ImageList images = get<0>(batch).to(device);
    vector<int64_t> image_ids = get<2>(batch);
    inference_timer.tic();
    output = model->forward(images);
    total += images.get_tensors().sum().item<int64_t>();
    first += images.get_tensors().select(1, 0).sum().item<int64_t>();
    second += images.get_tensors().select(1, 1).sum().item<int64_t>();
    third += images.get_tensors().select(1, 2).sum().item<int64_t>();
    inference_timer.toc();
    for(auto& i : output)
      i = i.To(cpu_device);
    assert(output.size() == image_ids.size());
    for(int i = 0; i < output.size(); ++i)
      results_map.insert({image_ids[i], output[i]});
    progress += images.get_tensors().size(0);
    std::cout << progress << "/" << total_size << "\n";

    torch::Tensor tmp = get<0>(batch).get_tensors();
    std::cout << tmp.size(0) << " " << output.size() << "\n";
  
    for(int i = 0; i < tmp.size(0); ++i){
      cv::Mat warp(tmp.size(2), tmp.size(3), CV_32FC3, tmp[i].permute({1, 2, 0}).contiguous().data<float>());

      if(output[i].Length() == 0)
        continue;
      // else
      //   std::cout << output[i].get_bbox() << "\n";
      for(int k = 0; k < output[i].Length(); ++k){
        cv::Point pt1(output[i].get_bbox()[k][0].item<int>(), output[i].get_bbox()[k][1].item<int>());
        // and its bottom right corner.
        cv::Point pt2(output[i].get_bbox()[k][2].item<int>(), output[i].get_bbox()[k][3].item<int>());
        cv::rectangle(warp, pt1, pt2, cv::Scalar(0, 255, 0));
      }
        
      
      cv::imwrite("../resource/tmp/" + std::to_string(image_ids[i]) + ".jpg", warp);
    }
    // if(progress > 120)
    //   break;
  }
  std::cout << total << "\n";
  std::cout << first << "\n";
  std::cout << second << "\n";
  std::cout << third << "\n";
  return results_map;
}

void inference();

}
}