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
#include "optimizer.h"

#include "datasets/coco_datasets.h"
#include "collate_batch.h"

#include <torch/script.h>
#include "bisect.h"
#include "trainer.h"

#include "inference.h"


using namespace std;
using namespace rcnn;
using namespace coco;

std::map<string, torch::Tensor> saved;

void tmp_recur(shared_ptr<torch::jit::script::Module> module, std::string name){
    string new_name;
    if(name.compare("") != 0)
      new_name = name + ".";
    
    for(auto& u : module->get_parameters()){
      torch::Tensor tensor = u.value().toTensor();
      saved[new_name + u.name()] = tensor;
    }
    for(auto& u : module->get_attributes()){
      torch::Tensor tensor = u.value().toTensor();
      saved[new_name + u.name()] = tensor;
    }
    for(auto& i : module->get_modules())
      tmp_recur(i, new_name + i->name());
  }

int main() {
  // torch::set_default_dtype(torch::kF32);
  rcnn::config::SetCFGFromFile("../resource/e2e_faster_rcnn_R_50_C4_1x.yaml");
  cout << "load complete" << endl;
  
  // engine::do_train();
  engine::inference();
  
  // torch::NoGradGuard guard;
  // auto body = modeling::ResNet();

  // auto module_ = torch::jit::load("../resource/resnet101.pth");
  // tmp_recur(module_, "");
  // for(auto i = saved.begin(); i != saved.end(); ++i)
  //   cout << i->first << "\n";
  
  // for(auto& i : body->named_parameters()){
  //   cout << i.key() << "\n";
  //   if(i.key().find("stem") != string::npos)
  //     i.value().copy_(saved.at(i.key().substr(5)));
  //   else
  //     i.value().copy_(saved.at(i.key()));
  // }
  // for(auto& i : body->named_buffers()){
  //   cout << i.key() << "\n";    
  //   if(i.key().find("stem") != string::npos)
  //     i.value().copy_(saved.at(i.key().substr(5)));
  //   else
  //     i.value().copy_(saved.at(i.key()));
  // }
  
  //   torch::serialize::OutputArchive archive;
  // for(auto& i : body->named_parameters())
  //   archive.write(i.key(), i.value());
  // for(auto& i : body->named_buffers()){
  //   std::cout << "saved buffer name : " << i.key() << "\n";
  //   archive.write(i.key(), i.value(), true);
  // }
  // archive.save_to("../resource/resnet101_cpp.pth");
  
  

  // auto model = modeling::BuildDetectionModel();
  // auto concat = solver::ConcatOptimizer(model->named_parameters(), 0.1, 0.1, 0.1);

  // auto cd = data::COCODataset("instances_val2017.json", "/val2017", false)
  //           .map(data::BatchCollator());;
  // auto test_loader =
  //     torch::data::make_data_loader(std::move(cd), 2);
  // for (const auto& batch : *test_loader){
  //   cout << get<0>(batch)[0].sizes() << "\n";
  // }
  //to image list
  // auto toimagetest_first = torch::randn({1, 3, 10, 10});
  // auto toimagetest_second = torch::randn({1, 3, 11, 11});
  // vector<torch::Tensor> vectoimagetest;
  // vectoimagetest.push_back(toimagetest_first);
  // vectoimagetest.push_back(toimagetest_second);
  // auto img_list = rcnn::structures::ToImageList(vectoimagetest, 3);
  // cout << img_list.get_tensors() << endl;
  //


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

  // auto cc = rcnn::data::COCODetection("/home/ocrusr/datasets/MSCOCO/val2017", "/home/ocrusr/datasets/MSCOCO/annotations/instances_val2017.json");
  // COCO cc = COCO("../resource/instances_val2017.json");
  // cout << "end coco load\n";
  // auto cc_ = cc.LoadRes("../resource/instances_val2014_fakesegm.json");

  // cout << cc_.anns[15].area << "\n";
  // cout << cc_.anns[15].id << "\n";
  // cout << cc_.anns[15].image_id << "\n";
  // cout << cc_.anns[15].compressed_rle << "\n";
  // auto data = cc.get(0);
  // std::vector<std::vector<std::vector<double>>> results;
  // // data.target.segmentation
  // // cout << data.data.sizes() << "\n";
  // for(auto& target : data.target)
  //   results.push_back(target.segmentation);
  
  // auto seg = rcnn::structures::SegmentationMask(results, make_pair(data.data.size(3), data.data.size(2)), "poly");
  // cout << "end seg \n";
  // cout <<seg.GetMaskTensor().sizes();
  //   cout << "========\n";
  // cout << data.target;
  return 0;
}