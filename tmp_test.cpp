#include <iostream>
#include <typeinfo>
#include <cassert>
#include "defaults.h"
#include "modeling.h"
#include "image_list.h"
#include <torch/torch.h>
#include "coco.h"
#include "tovec.h"
#include "coco_detection.h"
#include "segmentation_mask.h"


using namespace std;
using namespace rcnn;
using namespace coco;

int main() {
  // torch::set_default_dtype(torch::kF32);
  rcnn::config::SetCFGFromFile("/home/ocrusr/pytorch/maskrcnn_benchmark.cpp/e2e_faster_rcnn_R_50_C4_1x.yaml");
  cout << "load complete" << endl;
  auto model = rcnn::modeling::BuildDetectionModel();
  model->to(torch::kCUDA);
  auto input = torch::zeros({2, 3, 800, 800}).to(torch::kCUDA);
  std::vector<rcnn::structures::BoxList> model_results = model->forward(input);
  cout << model_results << "\n";
  // cout << z << "\n";
  // torch::nn::Sequential model;
  // {
  // rcnn::config::CFGS name = rcnn::config::GetCFG<rcnn::config::CFGS>({"MODEL", "BACKBONE", "CONV_BODY"});
  // std::string conv_body(name.get());
  // auto body = rcnn::modeling::ResNet(conv_body);
  // model->push_back(body);
  // }

  // torch::nn::Sequential m = rcnn::modeling::BuildBackbone();
  // for(auto& i: m->named_parameters())
  //   cout << i.key() << "\n";
  auto head = rcnn::modeling::MakeROIMaskPredictor(256);
  // cout << head << endl;
  // head->to(torch::kCUDA);
  // cout << m << endl;
  // auto t = torch::randn({0, 256, 15, 15}).cuda();
  // auto m = torch::randn({1, 256, 15, 15}).cuda();
  // cout << head->forward(t).sizes() << endl;
  // cout << head->forward(m).sizes() << endl;
  // cout << head->forward(t) << 
  // vector<torch::Tensor> results = m->forward<vector<torch::Tensor>>(t);
  // cout << results[0].sizes() << endl;
  // cout << results[1].sizes() << endl;
  // cout << results[2].sizes() << endl;
  // cout << results[3].sizes() << endl;
  // cout << results[4].sizes() << endl;


  //to image list
  // auto toimagetest_first = torch::randn({1, 3, 10, 10});
  // auto toimagetest_second = torch::randn({1, 3, 11, 11});
  // vector<torch::Tensor> vectoimagetest;
  // vectoimagetest.push_back(toimagetest_first);
  // vectoimagetest.push_back(toimagetest_second);
  // auto img_list = rcnn::structures::ToImageList(vectoimagetest, 3);
  // cout << img_list.get_tensors() << endl;
  //

  //////////////////anchor generator
  // int64_t stride = 16;
  // vector<int64_t> anchor_sizes{8, 16, 32};
  // vector<float> aspect_ratios{0.5, 1, 2};
  // auto result_anchor = rcnn::modeling::GenerateAnchors(stride, anchor_sizes, aspect_ratios);
  // cout << result_anchor << endl;

  //original implementation result
  // tensor([[  2.2500,   5.0000,  12.7500,  10.0000],
  //       [ -3.5000,   2.0000,  18.5000,  13.0000],
  //       [-15.0000,  -4.0000,  30.0000,  19.0000],
  //       [  4.0000,   4.0000,  11.0000,  11.0000],
  //       [  0.0000,   0.0000,  15.0000,  15.0000],
  //       [ -8.0000,  -8.0000,  23.0000,  23.0000],
  //       [  5.2500,   2.5000,   9.7500,  12.5000],
  //       [  2.5000,  -3.0000,  12.5000,  18.0000],
  //       [ -3.0000, -14.0000,  18.0000,  29.0000]], dtype=torch.float64)

  // vector<int64_t> feature_sizes{128, 256, 512};
  // rcnn::modeling::AnchorGenerator anchorclass = rcnn::modeling::MakeAnchorGenerator();
  // vector<torch::Tensor> tmp_features;
  // tmp_features.push_back(torch::randn({1, 1, 10, 10}));
  // // tmp_features.push_back(torch::randn({1, 1, 15, 15}));
  // anchorclass->forward(img_list, tmp_features);
  //////////////////

  //#####balanced sampler
  // auto sampler = rcnn::modeling::BalancedPositiveNegativeSampler(6, 0.5);
  // vector<torch::Tensor> vec;
  // auto first = torch::tensor({0, 0, 0, 0, 0, 0}, torch::TensorOptions().dtype(torch::kInt8));
  // vec.push_back(first);
  // auto second = torch::tensor({1, 1, 1, 0, 0, 0}, torch::TensorOptions().dtype(torch::kInt8));
  // vec.push_back(second);
  // auto third = torch::tensor({1, 1, 1, 1, 1, 1}, torch::TensorOptions().dtype(torch::kInt8));
  // vec.push_back(third);
  // auto result = sampler(vec);
  //######

  // cout << rcnn::config::GetCFG<std::vector<int>>({"MODEL", "ROI_MASK_HEAD", "CONV_LAYERS"})[0] <<endl;
  // cout << rcnn::config::GetCFG<bool>({"MODEL", "RPN_ONLY"}) <<endl;
  //Declare 3 dimension tensor with batch dimension
  // auto c = rcnn::layers::Conv2d(torch::nn::Conv2dOptions(3, 3, 3)
  //                  .stride(1)
  //                  .padding(1)
  //                  .with_bias(false));
  
//   auto c = rcnn::modeling::BuildBackBone();
//   cout << c << endl;
//   auto t = torch::randn({2, 3, 800, 800});
//   auto results = c->forward<deque<torch::Tensor>>(t);
  
  
  ////////////////bounding box
  //init bbox tensor size 2, 4
  // torch::Tensor box = torch::tensor({1, 1, 4, 4, 10, 10, 5, 5, 50, 50, 10, 10, 80, 80, 10, 10, 30, 30, 4, 4, 30, 30, 3, 3}).reshape({-1, 4}).to(torch::kF32);
  // torch::Tensor scores = torch::tensor({0.7, 0.6, 0.8, 0.9, 0.9, 0.5}).to(torch::kF32);
  // //init boxlist class
  // structures::BoxList bb = structures::BoxList(box, make_pair(100, 120), "xywh");
  // bb.AddField("scores", scores);
  // cout << bb.nms(0.5) << endl;
  // cout << bb.RemoveSmallBoxes(5) << endl;
  // cout << bb << endl;
  // //add label and score (dummy)
  // bb.AddField("labels", torch::tensor({1, 1}));
  // bb.AddField("scores", torch::tensor({0.4, 0.7}));
  // //print if is saved
  // auto scores = bb.GetField("scores");
  // cout << bb.GetField("scores") << endl;

  // //print fields
  // cout << "fields" << endl;
  // cout << bb.Fields() << endl;
  // //convert to xywh
  // auto xywh_bbox = bb.Convert("xywh");
  // cout << "converted bbox" << endl;
  // cout << xywh_bbox << endl;

  // //resize box
  // cout << "before resize" << endl;
  // cout << bb.get_bbox() << endl;
  // auto resized_bbox = bb.Resize(make_pair(200, 260));
  // cout << "after resize" << endl;
  // cout << resized_bbox.get_bbox() << endl;
  
  // //mask box with score
  // auto score_mask = scores > 0.5;
  // auto masked_bbox = bb[score_mask]; 
  // cout << "scores over 0.5 : " <<endl;
  // cout << masked_bbox << endl;
  // cout << masked_bbox.get_bbox() << endl;

  auto cc = rcnn::data::COCODetection("/home/ocrusr/datasets/MSCOCO/val2017", "/home/ocrusr/datasets/MSCOCO/annotations/instances_val2017.json");
  //COCO cc = COCO("/home/ocrusr/datasets/MSCOCO/annotations/instances_val2017.json");
  auto data = cc.get(0);
  std::vector<std::vector<std::vector<double>>> results;
  // data.target.segmentation
  // cout << data.data.sizes() << "\n";
  for(auto& target : data.target)
    results.push_back(target.segmentation);
  
  auto seg = rcnn::structures::SegmentationMask(results, make_pair(data.data.size(3), data.data.size(2)), "poly");
  cout << "end seg \n";
  cout <<seg.GetMaskTensor().sizes();
  //   cout << "========\n";
  // cout << data.target;
  return 0;
}