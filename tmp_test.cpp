#include <iostream>
#include <typeinfo>
#include <cassert>
#include "defaults.h"
#include "modeling.h"
#include <torch/torch.h>

#include <torch/script.h>
#include <jit_to_cpp.h>

#include "trainer.h"

#include "inference.h"


using namespace std;
using namespace rcnn;
using namespace coco;



int main() {

  // torch::set_default_dtype(torch::kF32);
  rcnn::config::SetCFGFromFile("../configs/e2e_faster_rcnn_R_101_FPN_1x.yaml");
  // cout << "load complete" << endl;
  //rcnn::utils::jit_to_cpp("../python_utils", "../configs/e2e_faster_rcnn_R_101_FPN_1x.yaml", std::vector<std::string>{"backbone.pth", "bbox_pred.pth", "cls_score.pth", "extractor_fc6.pth", "extractor_fc7.pth", "rpn_bbox.pth", "rpn_conv.pth", "rpn_logits.pth"});
  
  // // engine::do_train();
  engine::inference();

  return 0;
}