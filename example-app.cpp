#include <iostream>
#include "bounding_box.h"
#include "defaults.h"
#include "misc.h"


using namespace std;
using namespace rcnn;

int main() {
  //Declare 3 dimension tensor with batch dimension
  auto c = rcnn::layers::Conv2d(torch::nn::Conv2dOptions(3, 3, 3)
                   .stride(1)
                   .padding(1)
                   .with_bias(false));
  // auto conf = rcnn::config::GetDefaultCFG();
  // cout << conf << endl;
  //init bbox tensor size 2, 4
  torch::Tensor box = torch::tensor({1, 1, 4, 4, 10, 10, 50, 50}).reshape({2, 4});
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
}