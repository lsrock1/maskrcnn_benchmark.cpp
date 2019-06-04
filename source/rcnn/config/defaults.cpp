#include "defaults.h"
#include <vector>
#include <cstring>
#include <cassert>


namespace rcnn{
namespace config{

void SetCFGFromFile(const char*  file_path){
  if(cfg)
    return;
  cfg = new YAML::Node(YAML::LoadFile(file_path));
  
  SetNode((*cfg)["MODEL"], YAML::Node());
  SetNode((*cfg)["MODEL"]["RPN_ONLY"], false);
  SetNode((*cfg)["MODEL"]["RETINANET_ON"], false);
  SetNode((*cfg)["MODEL"]["KEYPOINT_ON"], false);
  SetNode((*cfg)["MODEL"]["DEVICE"], "cuda");
  SetNode((*cfg)["MODEL"]["META_ARCHITECTURE"], "GeneralizedRCNN");
  SetNode((*cfg)["MODEL"]["CLS_AGNOSTIC_BBOX_REG"], false);
  SetNode((*cfg)["MODEL"]["WEIGHT"], "");
  //input
  SetNode((*cfg)["INPUT"], YAML::Node());
  SetNode((*cfg)["INPUT"]["MIN_SIZE_TRAIN"], 800);
  SetNode((*cfg)["INPUT"]["MAX_SIZE_TRAIN"], 1333);
  SetNode((*cfg)["INPUT"]["MIN_SIZE_TEST"], 800);
  SetNode((*cfg)["INPUT"]["MAX_SIZE_TEST"], 1333);
  SetNode((*cfg)["INPUT"]["PIXEL_MEAN"], "(102.9801, 115.9465, 122.7717)");
  SetNode((*cfg)["INPUT"]["PIXEL_STD"], "(1., 1., 1.)");
  SetNode((*cfg)["INPUT"]["TO_BGR255"], true);

  SetNode((*cfg)["INPUT"]["RIGHTNESS"], 0.0);
  SetNode((*cfg)["INPUT"]["CONTRAST"], 0.0);
  SetNode((*cfg)["INPUT"]["SATURATION"], 0.0);
  SetNode((*cfg)["INPUT"]["HUE"], 0.0);

  //DATASET
  SetNode((*cfg)["DATASETS"], YAML::Node());
  SetNode((*cfg)["DATASETS"]["TRAIN"], "(COCO, )");
  SetNode((*cfg)["DATASETS"]["TEST"], "(COCO, )");
  //DATALOADER
  SetNode((*cfg)["DATALOADER"], YAML::Node());
  SetNode((*cfg)["DATALOADER"]["NUM_WORKERS"], 4);
  SetNode((*cfg)["DATALOADER"]["SIZE_DIVISIBILITY"], 0);
  SetNode((*cfg)["DATALOADER"]["ASPECT_RATIO_GROUPING"], true);
  
  //BACKBONE
  SetNode((*cfg)["MODEL"]["BACKBONE"], YAML::Node());
  SetNode((*cfg)["MODEL"]["BACKBONE"]["CONV_BODY"], "R-50-C4");
  SetNode((*cfg)["MODEL"]["BACKBONE"]["FREEZE_CONV_BODY_AT"], 2);
  //SetNode((*cfg)["MODEL"]["BACKBONE"]["USE_GN"], false);
  
  //FPN
  SetNode((*cfg)["MODEL"]["FPN"], YAML::Node());
  //SetNode((*cfg)["MODEL"]["FPN"]["USE_GN"], false);
  SetNode((*cfg)["MODEL"]["FPN"]["USE_RELU"], false);
  

  //Group Norm
//   SetNode((*cfg)["MODEL"]["GROUP_NORM"], YAML::Node());
//   SetNode((*cfg)["MODEL"]["GROUP_NORM"]["DIM_PER_GP"], -1);
//   SetNode((*cfg)["MODEL"]["GROUP_NORM"]["NUM_GROUPS"], 32);
//   SetNode((*cfg)["MODEL"]["GROUP_NORM"]["EPSILON"], 1e-5);
  //RPN
  SetNode((*cfg)["MODEL"]["RPN"], YAML::Node());
  SetNode((*cfg)["MODEL"]["RPN"]["USE_FPN"], false);
  SetNode((*cfg)["MODEL"]["RPN"]["ANCHOR_SIZES"], "(32, 64, 128, 256, 512)");
  SetNode((*cfg)["MODEL"]["RPN"]["ANCHOR_STRIDE"], "(16,)");
  SetNode((*cfg)["MODEL"]["RPN"]["ASPECT_RATIOS"], "(0.5, 1.0, 2.0)");
  SetNode((*cfg)["MODEL"]["RPN"]["STRADDLE_THRESH"], 0);
  SetNode((*cfg)["MODEL"]["RPN"]["FG_IOU_THRESHOLD"], 0.7);
  SetNode((*cfg)["MODEL"]["RPN"]["BG_IOU_THRESHOLD"], 0.3);
  SetNode((*cfg)["MODEL"]["RPN"]["BATCH_SIZE_PER_IMAGE"], 256);
  SetNode((*cfg)["MODEL"]["RPN"]["POSITIVE_FRACTION"], 0.5);
  SetNode((*cfg)["MODEL"]["RPN"]["PRE_NMS_TOP_N_TRAIN"], 12000);
  SetNode((*cfg)["MODEL"]["RPN"]["PRE_NMS_TOP_N_TEST"], 6000);
  SetNode((*cfg)["MODEL"]["RPN"]["POST_NMS_TOP_N_TRAIN"], 2000);
  SetNode((*cfg)["MODEL"]["RPN"]["POST_NMS_TOP_N_TEST"], 1000);
  SetNode((*cfg)["MODEL"]["RPN"]["NMS_THRESH"], 0.7);
  SetNode((*cfg)["MODEL"]["RPN"]["MIN_SIZE"], 0);
  SetNode((*cfg)["MODEL"]["RPN"]["FPN_POST_NMS_TOP_N_TRAIN"], 2000);
  SetNode((*cfg)["MODEL"]["RPN"]["FPN_POST_NMS_TOP_N_TEST"], 2000);
  SetNode((*cfg)["MODEL"]["RPN"]["FPN_POST_NMS_PER_BATCH"], true);
  SetNode((*cfg)["MODEL"]["RPN"]["RPN_HEAD"], "SingleConvRPNHead");

  //ROI HEAD
  SetNode((*cfg)["MODEL"]["ROI_HEADS"], YAML::Node());
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["USE_FPN"], false);
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["FG_IOU_THRESHOLD"], 0.5);
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["BG_IOU_THRESHOLD"], 0.5);
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["BBOX_REG_WEIGHTS"], "(10., 10., 5., 5.)");
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["BATCH_SIZE_PER_IMAGE"], 512);
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["POSITIVE_FRACTION"], 0.25);
  //only used on test mode
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["SCORE_THRESH"], 0.05);
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["NMS"], 0.5);
  SetNode((*cfg)["MODEL"]["ROI_HEADS"]["DETECTIONS_PER_IMG"], 100);
  
  //ROI BOX HEAD
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"], YAML::Node());
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["FEATURE_EXTRACTOR"], "ResNet50Conv5ROIFeatureExtractor");
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["PREDICTOR"], "FastRCNNPredictor");
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["POOLER_RESOLUTION"], 14);
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["POOLER_SAMPLING_RATIO"], 0);
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["POOLER_SCALES"], "(1.0 / 16.,)");
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"], 81);
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["MLP_HEAD_DIM"], 1024);
  // SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["USE_GN"], false);
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["DILATION"], 1);
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["CONV_HEAD_DIM"], 256);
  SetNode((*cfg)["MODEL"]["ROI_BOX_HEAD"]["NUM_STACKED_CONVS"], 4);

  //ROI MASK HEAD
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"], YAML::Node());
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["FEATURE_EXTRACTOR"], "ResNet50Conv5ROIFeatureExtractor");
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["PREDICTOR"], "MaskRCNNC4Predictor");
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["POOLER_RESOLUTION"], 14);
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["POOLER_SAMPLING_RATIO"], 0);
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["POOLER_SCALES"], "(1.0 / 16.,)");
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["MLP_HEAD_DIM"], 1024);
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["CONV_LAYERS"], "(256, 256, 256, 256)");
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["RESOLUTION"], 14);
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["SHARE_BOX_FEATURE_EXTRACTOR"], true);
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["POSTPROCESS_MASKS"], false);
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["POSTPROCESS_MASKS_THRESHOLD"], 0.5);
  SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["DILATION"], 1);
  //SetNode((*cfg)["MODEL"]["ROI_MASK_HEAD"]["USE_GN"], false);

  //RESNET
  SetNode((*cfg)["MODEL"]["RESNETS"], YAML::Node());
  SetNode((*cfg)["MODEL"]["RESNETS"]["NUM_GROUPS"], 1);
  SetNode((*cfg)["MODEL"]["RESNETS"]["WIDTH_PER_GROUP"], 64);
  SetNode((*cfg)["MODEL"]["RESNETS"]["STRIDE_IN_1X1"], true);
  SetNode((*cfg)["MODEL"]["RESNETS"]["TRANS_FUNC"], "BottleneckWithFixedBatchNorm");
  SetNode((*cfg)["MODEL"]["RESNETS"]["STEM_FUNC"], "StemWithFixedBatchNorm");
  SetNode((*cfg)["MODEL"]["RESNETS"]["RES5_DILATION"], 1);
  SetNode((*cfg)["MODEL"]["RESNETS"]["BACKBONE_OUT_CHANNELS"], 1024);
  SetNode((*cfg)["MODEL"]["RESNETS"]["RES2_OUT_CHANNELS"], 256);
  SetNode((*cfg)["MODEL"]["RESNETS"]["STEM_OUT_CHANNELS"], 64);
  // std::vector<bool> vec{false, false, false, false};
  // SetNode((*cfg)["MODEL"]["RESNET"]["STAGE_WITH_DCN"], vec);
  // SetNode((*cfg)["MODEL"]["RESNET"]["DEFORMABLE_GROUPS"], 1);

  //SOLVER
  SetNode((*cfg)["SOLVER"], YAML::Node());
  SetNode((*cfg)["SOLVER"]["MAX_ITER"], 40000);
  SetNode((*cfg)["SOLVER"]["BASE_LR"], 0.001);
  SetNode((*cfg)["SOLVER"]["BASE_LR_FACTOR"], 2);
  SetNode((*cfg)["SOLVER"]["MOMENTUM"], 0.9);
  SetNode((*cfg)["SOLVER"]["WEIGHT_DECAY"], 0.0005);
  SetNode((*cfg)["SOLVER"]["WEIGHT_DECAY_BIAS"], 0);
  SetNode((*cfg)["SOLVER"]["GAMMA"], 0.1);
  SetNode((*cfg)["SOLVER"]["STEPS"], "(30000, )");
  SetNode((*cfg)["SOLVER"]["WARMUP_FACTOR"], 1.0 / 3.0);
  SetNode((*cfg)["SOLVER"]["WARMUP_ITERS"], 500);
  SetNode((*cfg)["SOLVER"]["WARMUP_METHOD"], "linear");
  SetNode((*cfg)["SOLVER"]["CHECKPOINT_PERIOD"], 2500);
  SetNode((*cfg)["SOLVER"]["IMS_PER_BATCH"], 16);

  //TEST
  SetNode((*cfg)["TEST"], YAML::Node());
  SetNode((*cfg)["TEST"]["EXPECTED_RESULTS"], "(,)");
  SetNode((*cfg)["TEST"]["EXPECTED_RESULTS_SIGMA_TOL"], 4);
  SetNode((*cfg)["TEST"]["IMS_PER_BATCH"], 8);
  SetNode((*cfg)["TEST"]["DETECTIONS_PER_IMG"], 100);

  SetNode((*cfg)["TEST"]["BBOX_AUG"], YAML::Node());
  SetNode((*cfg)["TEST"]["BBOX_AUG"]["ENABLED"], false);
  SetNode((*cfg)["TEST"]["BBOX_AUG"]["H_FLIP"], false);
  SetNode((*cfg)["TEST"]["BBOX_AUG"]["SCALES"], "(,)");
  SetNode((*cfg)["TEST"]["BBOX_AUG"]["MAX_SIZE"], 4000);
  SetNode((*cfg)["TEST"]["BBOX_AUG"]["SCALE_H_FLIP"], false);
  

  //MISC OPTIONS
  SetNode((*cfg)["OUTPUT_DIR"], ".");
  //default_config["PATH_CATALOG"], solver;
  SetNode((*cfg)["DTYPE"], "float32");
}

CFGS::CFGS(std::string result)
          :name_(result),
          name_c_(strdup(result.c_str())){};

CFGS::~CFGS(){
  delete[] name_c_;
}

const char* CFGS::get(){
  return name_c_;
}

template<typename T>
T GetCFG(std::initializer_list<const char*> node){
  if(!cfg){
    throw "Set Config file first";
  }
  std::vector<YAML::Node> tmp{*cfg};
  for(auto i = node.begin(); i != node.end(); ++i){
    tmp.push_back(tmp.back()[*i]);
  }
  return tmp.back().as<T>();
}

template bool GetCFG<bool>(std::initializer_list<const char*> node);
template int64_t GetCFG<int64_t>(std::initializer_list<const char*> node);
template int GetCFG<int>(std::initializer_list<const char*> node);
template float GetCFG<float>(std::initializer_list<const char*> node);

template<>
CFGS GetCFG<CFGS>(std::initializer_list<const char*> node){
  if(!cfg){
    throw "Set Config file first";
  }
  std::vector<YAML::Node> tmp{*cfg};
  for(auto i = node.begin(); i != node.end(); ++i){
    tmp.push_back(tmp.back()[*i]);
  }
  return CFGS(tmp.back().as<std::string>());
}

template<>
std::vector<float> GetCFG<std::vector<float>>(std::initializer_list<const char*> node){
  if(!cfg){
    throw "Set Config file first";
  }
  std::vector<YAML::Node> tmp{*cfg};
  for(auto i = node.begin(); i != node.end(); ++i){
    tmp.push_back(tmp.back()[*i]);
  }
  std::string svalue = tmp.back().as<std::string>();
  std::vector<float> elements;
  size_t pos = 0;
  //remove white spaces
  std::string::iterator end_pos = std::remove(svalue.begin(), svalue.end(), ' ');
  svalue.erase(end_pos, svalue.end());
  //remove ( and )
  svalue = svalue.substr(1, svalue.size()-2);
  std::string token;
  while ((pos = svalue.find(",")) != std::string::npos) {
    token = svalue.substr(0, pos);
    elements.push_back(std::stof(token));
    svalue.erase(0, pos + 1);
  }
  if(svalue.size() > 1){
    elements.push_back(std::stof(svalue));
  }
  return elements;
}

template<>
std::vector<int64_t> GetCFG<std::vector<int64_t>>(std::initializer_list<const char*> node){
  if(!cfg){
    throw "Set Config file first";
  }
  std::vector<YAML::Node> tmp{*cfg};
  for(auto i = node.begin(); i != node.end(); ++i){
    tmp.push_back(tmp.back()[*i]);
  }
  std::string svalue = tmp.back().as<std::string>();
  std::vector<int64_t> elements;
  size_t pos = 0;
  //remove white spaces
  std::string::iterator end_pos = std::remove(svalue.begin(), svalue.end(), ' ');
  svalue.erase(end_pos, svalue.end());
  //remove ( and )
  svalue = svalue.substr(1, svalue.size()-2);
  
  std::string token;
  while ((pos = svalue.find(",")) != std::string::npos) {
    token = svalue.substr(0, pos);
    elements.push_back(std::stoi(token));
    svalue.erase(0, pos + 1);
  }
  if(svalue.size() > 1){
    elements.push_back(std::stoi(svalue));
  }
  return elements;
}

}//mrcn
}//configs