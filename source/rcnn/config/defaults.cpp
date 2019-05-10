#include "defaults.h"
#include "yaml-cpp/yaml.h"

namespace rcnn{
namespace config{

void SetDefaultCFGFromFile(std::string file_path){

}

const YAML::Node* GetDefaultCFG(){
  if(cfg){
    return cfg;
  }
  else{
    cfg = new YAML::Node();
    
    (*cfg)["MODEL"] = YAML::Node();
    (*cfg)["MODEL"]["RPN_ONLY"] = false;
    (*cfg)["MODEL"]["MASK_ON"] = false;
    (*cfg)["MODEL"]["RETINANET_ON"] = false;
    (*cfg)["MODEL"]["KEYPOINT_ON"] = false;
    (*cfg)["MODEL"]["DEVICE"] = "cuda";
    (*cfg)["MODEL"]["META_ARCHITECTURE"] = "GeneralizedRCNN";
    (*cfg)["MODEL"]["CLS_AGNOSTIC_BBOX_REG"] = false;
    (*cfg)["MODEL"]["WEIGHT"] = "";
    //input
    YAML::Node input;
    input["MIN_SIZE_TRAIN"] = 800;
    input["MAX_SIZE_TRAIN"] = 1333;
    input["MIN_SIZE_TEST"] = 800;
    input["MAX_SIZE_TEST"] = 1333;
    std::vector<double> pixel_mean{102.9801, 115.9465, 122.7717};
    input["PIXEL_MEAN"] = pixel_mean;
    std::vector<double> pixel_std{1., 1., 1.};
    input["PIXEL_STD"] = pixel_std;
    input["TO_BGR255"] = true;

    input["RIGHTNESS"] = 0.0;
    input["CONTRAST"] = 0.0;
    input["SATURATION"] = 0.0;
    input["HUE"] = 0.0;
    (*cfg)["INPUT"] = input;

    //DATASET
    YAML::Node dataset;
    std::vector<std::string> dataset_train{"COCO"};
    dataset["TRAIN"] = dataset_train;
    std::vector<std::string> dataset_test{"COCO"};
    dataset["TEST"] = dataset_test;
    (*cfg)["DATASET"] = dataset;
    //DATALOADER
    YAML::Node dataloader;
    dataloader["NUM_WORKERS"] = 4;
    dataloader["SIZE_DIVISIBILITY"] = 0;
    dataloader["ASPECT_RATIO_GROUPING"] = true;
    (*cfg)["DATALOADER"] = dataloader;
      
    //BACKBONE
    YAML::Node backbone;
    backbone["CONV_BODY"] = "R-50-C4";
    backbone["FREEZE_CONV_BODY_AT"] = 2;
    //backbone["USE_GN"] = false;
    (*cfg)["BACKBONE"] = backbone;

    //FPN
    YAML::Node fpn;
    //fpn["USE_GN"] = false;
    fpn["USE_RELU"] = false;
    (*cfg)["FPN"] = fpn;

    //Group Norm
  //   YAML::Node group_norm;
  //   group_norm["DIM_PER_GP"] = -1;
  //   group_norm["NUM_GROUPS"] = 32;
  //   group_norm["EPSILON"] = 1e-5;
    //RPN
    YAML::Node rpn;
    rpn["USE_FPN"] = false;
    std::vector<int> anchor_size{32, 64, 128, 256, 512};
    rpn["ANCHOR_SIZE"] = anchor_size;
    std::vector<int> anchor_stride{16};
    rpn["ANCHOR_STRIDE"] = anchor_stride;
    std::vector<double> aspect_ratio{0.5, 1.0, 2.0};
    rpn["ASPECT_RATIO"] = aspect_ratio;
    rpn["STRADDLE_THRESH"] = 0;
    rpn["FG_IOU_THRESHOLD"] = 0.7;
    rpn["BG_IOU_THRESHOLD"] = 0.3;
    rpn["BATCH_SIZE_PER_IMAGE"] = 256;
    rpn["POSITIVE_FRACTION"] = 0.5;
    rpn["PRE_NMS_TOP_N_TRAIN"] = 12000;
    rpn["PRE_NMS_TOP_N_TEST"] = 6000;
    rpn["POST_NMS_TOP_N_TRAIN"] = 2000;
    rpn["POST_NMS_TOP_N_TEST"] = 1000;
    rpn["NMS_THRESH"] = 0.7;
    rpn["MIN_SIZE"] = 0;
    rpn["FPN_POST_NMS_TOP_N_TRAIN"] = 2000;
    rpn["FPN_POST_NMS_TOP_N_TEST"] = 2000;
    rpn["FPN_POST_NMS_PER_BATCH"] = true;
    rpn["RPN_HEAD"] = "SingleConvRPNHead";
    (*cfg)["RPN"] = rpn;

    //ROI HEAD
    YAML::Node roi_heads;
    roi_heads["USE_FPN"] = false;
    roi_heads["FG_IOU_THRESHOLD"] = 0.5;
    roi_heads["BG_IOU_THRESHOLD"] = 0.5;
    std::vector<double> bbox_reg_weights{10, 10., 5., 5.};
    roi_heads["BBOX_REG_WEIGHTS"] = bbox_reg_weights;
    roi_heads["BATCH_SIZE_PER_IMAGE"] = 512;
    roi_heads["POSITIVE_FRACTION"] = 0.25;
    //only used on test mode
    roi_heads["SCORE_THRESH"] = 0.05;
    roi_heads["NMS"] = 0.5;
    roi_heads["DETECTIONS_PER_IMG"] = 100;
    (*cfg)["ROI_HEADS"] = roi_heads;

    //ROI BOX HEAD
    YAML::Node roi_box_head;
    roi_box_head["FEATURE_EXTRACTOR"] = "ResNet50Conv5ROIFeatureExtractor";
    roi_box_head["PREDICTOR"] = "FastRCNNPredictor";
    roi_box_head["POOLER_RESOLUTION"] = 14;
    roi_box_head["POOLER_SAMPLING_RATIO"] = 0;
    roi_box_head["POOLER_SCALES"] = 1.0 / 16.;
    roi_box_head["NUM_CLASSES"] = 81;
    roi_box_head["MLP_HEAD_DIM"] = 1024;
    //roi_box_head["USE_GN"] = false;
    roi_box_head["DILATION"] = 1;
    roi_box_head["CONV_HEAD_DIM"] = 256;
    roi_box_head["NUM_STACKED_CONVS"] = 4;
    (*cfg)["ROI_BOX_HEAD"] = roi_box_head;

    //ROI MASK HEAD
    YAML::Node roi_mask_head;
    roi_mask_head["FEATURE_EXTRACTOR"] = "ResNet50Conv5ROIFeatureExtractor";
    roi_mask_head["PREDICTOR"] = "MaskRCNNC4Predictor";
    roi_mask_head["POOLER_RESOLUTION"] = 14;
    roi_mask_head["POOLER_SAMPLING_RATIO"] = 0;
    roi_mask_head["POOLER_SCALES"] = 1.0 / 16.;
    roi_mask_head["MLP_HEAD_DIM"] = 1024;
    std::vector<int> conv_layers{256, 256, 256, 256};
    roi_mask_head["CONV_LAYERS"] = conv_layers;
    roi_mask_head["RESOLUTION"] = 14;
    roi_mask_head["SHARE_BOX_FEATURE_EXTRACTOR"] = true;
    roi_mask_head["POSTPROCESS_MASKS"] = false;
    roi_mask_head["POSTPROCESS_MASKS_THRESHOLD"] = 0.5;
    roi_mask_head["DILATION"] = 1;
    //roi_mask_head["USE_GN"] = false;
    (*cfg)["ROI_MASK_HEAD"] = roi_mask_head;

    //RESNET
    YAML::Node resnet;
    resnet["NUM_GROUPS"] = 1;
    resnet["WIDTH_PER_GROUP"] = 64;
    resnet["STRIDE_IN_1X1"] = true;
    resnet["TRANS_FUNC"] = "BottleneckWithFixedBatchNorm";
    resnet["STEM_FUNC"] = "StemWithFixedBatchNorm";
    resnet["RES5_DILATION"] = 1;
    resnet["BACKBONE_OUT_CHANNELS"] = 256 * 4;
    resnet["RES2_OUT_CHANNELS"] = 256;
    resnet["STEM_OUT_CHANNELS"] = 64;
    // std::vector<bool> vec{false, false, false, false};
    // resnet["STAGE_WITH_DCN"] = vec;
    resnet["DEFORMABLE_GROUPS"] = 1;
    (*cfg)["RESNET"] = resnet;

    //SOLVER
    YAML::Node solver;
    solver["MAX_ITER"] = 40000;
    solver["BASE_LR"] = 0.001;
    solver["BASE_LR_FACTOR"] = 2;
    solver["MOMENTUM"] = 0.9;
    solver["WEIGHT_DECAY"] = 0.0005;
    solver["WEIGHT_DECAY_BIAS"] = 0;
    solver["GAMMA"] = 0.1;
    std::vector<int> steps{30000};
    solver["STEPS"] = steps;
    solver["WARMUP_FACTOR"] = 1.0 / 3.0;
    solver["WARMUP_ITERS"] = 500;
    solver["WARMUP_METHOD"] = "linear";
    solver["CHECKPOINT_PERIOD"] = 2500;
    solver["IMS_PER_BATCH"] = 16;
    (*cfg)["SOLVER"] = solver;

    //TEST
    YAML::Node test;
    std::vector<int> expected_results;
    test["EXPECTED_RESULTS"] = expected_results;
    test["EXPECTED_RESULTS_SIGMA_TOL"] = 4;
    test["IMS_PER_BATCH"] = 8;
    test["DETECTIONS_PER_IMG"] = 100;
    (*cfg)["TEST"] = test;

    //MISC OPTIONS
    (*cfg)["OUTPUT_DIR"] = ".";
    //default_config["PATH_CATALOG"] = solver;
    (*cfg)["DTYPE"] = "float32";
    return cfg;
  }
}

// YAML::Node SetConfigFromFile(std::string path){

// }
}//mrcn
}//configs