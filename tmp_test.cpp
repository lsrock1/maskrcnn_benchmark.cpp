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



int main() {
  // torch::set_default_dtype(torch::kF32);
  rcnn::config::SetCFGFromFile("../resource/e2e_faster_rcnn_R_50_FPN_1x.yaml");
  cout << "load complete" << endl;
  
  // engine::do_train();
  engine::inference();
  return 0;
}