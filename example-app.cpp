#include <iostream>
#include <typeinfo>
#include "defaults.h"
#include "backbone.h"
#include <torch/torch.h>
#include "balanced_positive_negative_sampler.h"


using namespace std;
using namespace rcnn;

int main() {
  rcnn::config::SetCFGFromFile("/root/e2e_faster_rcnn_R_50_FPN_1x.yaml");

  //#####balanced sampler
  // auto sampler = rcnn::modeling::BalancedPositiveNegativeSampler(4, 0.7);
  // vector<torch::Tensor> vec;
  // auto first = torch::randint(3, {8}, torch::TensorOptions().dtype(torch::kInt8));
  // vec.push_back(first);
  // auto second = torch::randint(3, {9}, torch::TensorOptions().dtype(torch::kInt8));
  // vec.push_back(second);
  // cout << "first : " << endl << first << endl;
  // cout << "seoncd : " << endl << second << endl;
  // auto result = sampler(vec);
  // cout << "end sampling" << endl;
  // cout << result.first << endl;
  // cout << result.second << endl;
  //######

  // cout << rcnn::config::GetCFG<std::vector<int>>({"MODEL", "ROI_MASK_HEAD", "CONV_LAYERS"})[0] <<endl;
  // cout << rcnn::config::GetCFG<bool>({"MODEL", "RPN_ONLY"}) <<endl;
  //Declare 3 dimension tensor with batch dimension
  // auto c = rcnn::layers::Conv2d(torch::nn::Conv2dOptions(3, 3, 3)
  //                  .stride(1)
  //                  .padding(1)
  //                  .with_bias(false));
  
  auto c = rcnn::modeling::BuildBackBone();
  cout << c << endl;
  auto t = torch::randn({2, 3, 800, 800});
  auto results = c->forward<deque<torch::Tensor>>(t);
  cout << results[0].sizes() << endl;
    cout << results[1].sizes() << endl;
  cout << results[2].sizes() << endl;
  cout << results[3].sizes() << endl;
  cout << results[4].sizes() << endl;
  // YAML::Node* conf2 = rcnn::config::GetDefaultCFG();
  // cout << (*conf2)["MODEL"] << endl;
  // cout << (*conf)["MODEL"]<< endl;
  // for(auto i = (*conf).begin(); i != (*conf).end(); ++i){
  //   cout << i->second << endl;
  // }
  // for(auto i = conf.begin(); i != conf.end(); ++i){
  //   cout << i->as<string>() << endl;
  // }
  // cout << conf << endl;
  //init bbox tensor size 2, 4
  // torch::Tensor box = torch::tensor({1, 1, 4, 4, 10, 10, 50, 50}).reshape({2, 4});
  // //init boxlist class
  // structures::BoxList bb = structures::BoxList(box, make_pair(100, 120), "xyxy");
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
  return 0;
}