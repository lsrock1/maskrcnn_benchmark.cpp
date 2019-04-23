#include <config/defaults.h>

namespace mrcn{
namespace configs{

CFG GetDefaultCFG(){
  CFG default_config = CFG();

  CFG model = CFG();
  model.SetDatum("RPN_ONLY", false);
  model.SetDatum("MASK_ON", false);
  model.SetDatum("RETINANET_ON", false);
  model.SetDatum("KEYPOINT_ON", false);
  model.SetDatum("DEVICE", "cuda");
  model.SetDatum("META_ARCHITECTURE", "GeneralizedRCNN");
  model.SetDatum("CLS_AGNOSTIC_BBOX_REG", false);
  model.SetDatum("WEIGHT", "");
  model.SetDatum("");
  default_config.SetDatum("MODEL", model);
  //input
  CFG input = CFG();
  input.SetDatum("MIN_SIZE_TRAIN", 800);
  input.SetDatum("MAX_SIZE_TRAIN", 1333);
  input.SetDatum("MIN_SIZE_TEST", 800);
  input.SetDatum("MAX_SIZE_TEST", 1333);
  input.SetDatum("PIXEL_MEAN", {102.9801, 115.9465, 122.7717});
  input.SetDatum("PIXEL_STD", {1., 1., 1.});
  input.SetDatum("TO_BGR255", true);

  input.SetDatum("RIGHTNESS", 0.0);
  input.SetDatum("CONTRAST", 0.0);
  input.SetDatum("SATURATION", 0.0);
  input.SetDatum("HUE", 0.0);
  default_config.SetDatum("INPUT", input);

  //DATASET
  CFG dataset = CFG();
  dataset.SetDatum("TRAIN", {});
  dataset.SetDatum("TEST", {});
  default_config.SetDatum("DATASET", dataset);
  //DATALOADER
  CFG dataloader = CFG();
  dataloader.SetDatum("NUM_WORKERS", 4);
  dataloader.SetDatum("SIZE_DIVISIBILITY", 0);
  dataloader.SetDatum("ASPECT_RATIO_GROUPING", true);
  default_config.SetDatum("DATALOADER", dataloader);
    
  //BACKBONE
  CFG backbone = CFG();
  backbone.SetDatum("CONV_BODY", "R-50-C4");
  backbone.SetDatum("FREEZE_CONV_BODY_AT", 2);
  //backbone.SetDatum("USE_GN", false);
  default_config.SetDatum("BACKBONE", backbone);

  //FPN
  CFG fpn = CFG();
  //fpn.SetDatum("USE_GN", false);
  fpn.SetDatum("USE_RELU", false);
  default_config.SetDatum("FPN", fpn);

  //Group Norm
//   CFG group_norm = CFG();
//   group_norm.SetDatum("DIM_PER_GP", -1);
//   group_norm.SetDatum("NUM_GROUPS", 32);
//   group_norm.SetDatum("EPSILON", 1e-5);
  //RPN
  CFG rpn = CFG();
  rpn.SetDatum("USE_FPN", false);
  rpn.SetDatum("ANCHOR_SIZE", {32, 64, 128, 256, 512});
  rpn.SetDatum("ANCHOR_STRIDE", {16});
  rpn.SetDatum("ASPECT_RATIO", {0.5, 1.0, 2.0});
  rpn.SetDatum("STRADDLE_THRESH", 0);
  rpn.SetDatum("FG_IOU_THRESHOLD", 0.7);
  rpn.SetDatum("BG_IOU_THRESHOLD", 0.3);
  rpn.SetDatum("BATCH_SIZE_PER_IMAGE", 256);
  rpn.SetDatum("POSITIVE_FRACTION", 0.5);
  rpn.SetDatum("PRE_NMS_TOP_N_TRAIN", 12000);
  rpn.SetDatum("PRE_NMS_TOP_N_TEST", 6000);
  rpn.SetDatum("POST_NMS_TOP_N_TRAIN", 2000);
  rpn.SetDatum("POST_NMS_TOP_N_TEST", 1000);
  rpn.SetDatum("NMS_THRESH", 0.7);
  rpn.SetDatum("MIN_SIZE", 0);
  rpn.SetDatum("FPN_POST_NMS_TOP_N_TRAIN", 2000);
  rpn.SetDatum("FPN_POST_NMS_TOP_N_TEST", 2000);
  rpn.SetDatum("FPN_POST_NMS_PER_BATCH", true);
  rpn.SetDatum("RPN_HEAD", "SingleConvRPNHead");
  default_config.SetDatum("RPN", rpn);

  //ROI HEAD
  CFG roi_heads = CFG();
  roi_heads.SetDatum("USE_FPN", false);
  roi_heads.SetDatum("FG_IOU_THRESHOLD", 0.5);
  roi_heads.SetDatum("BG_IOU_THRESHOLD", 0.5);
  roi_heads.SetDatum("BBOX_REG_WEIGHTS", {10., 10., 5., 5.});
  roi_heads.SetDatum("BATCH_SIZE_PER_IMAGE", 512);
  roi_heads.SetDatum("POSITIVE_FRACTION", 0.25);
  //only used on test mode
  roi_heads.SetDatum("SCORE_THRESH", 0.05);
  roi_heads.SetDatum("NMS", 0.5);
  roi_heads.SetDatum("DETECTIONS_PER_IMG", 100);
  default_config.SetDatum("ROI_HEADS", roi_heads);

  //ROI BOX HEAD
  CFG roi_box_head = CFG();
  roi_box_head.SetDatum("FEATURE_EXTRACTOR", "ResNet50Conv5ROIFeatureExtractor");
  roi_box_head.SetDatum("PREDICTOR", "FastRCNNPredictor");
  roi_box_head.SetDatum("POOLER_RESOLUTION", 14);
  roi_box_head.SetDatum("POOLER_SAMPLING_RATIO", 0);
  roi_box_head.SetDatum("POOLER_SCALES", 1.0 / 16.);
  roi_box_head.SetDatum("NUM_CLASSES", 81);
  roi_box_head.SetDatum("MLP_HEAD_DIM", 1024);
  //roi_box_head.SetDatum("USE_GN", false);
  roi_box_head.SetDatum("DILATION", 1);
  roi_box_head.SetDatum("CONV_HEAD_DIM", 256);
  roi_box_head.SetDatum("NUM_STACKED_CONVS", 4);
  default_config.SetDatum("ROI_BOX_HEAD", roi_box_head);

  //ROI MASK HEAD
  CFG roi_mask_head = CFG();
  roi_mask_head.SetDatum("FEATURE_EXTRACTOR", "ResNet50Conv5ROIFeatureExtractor");
  roi_mask_head.SetDatum("PREDICTOR", "MaskRCNNC4Predictor");
  roi_mask_head.SetDatum("POOLER_RESOLUTION", 14);
  roi_mask_head.SetDatum("POOLER_SAMPLING_RATIO", 0);
  roi_mask_head.SetDatum("POOLER_SCALES", 1.0 / 16.);
  roi_mask_head.SetDatum("MLP_HEAD_DIM", 1024);
  roi_mask_head.SetDatum("CONV_LAYERS", {256, 256, 256, 256});
  roi_mask_head.SetDatum("RESOLUTION", 14);
  roi_mask_head.SetDatum("SHARE_BOX_FEATURE_EXTRACTOR", true);
  roi_mask_head.SetDatum("POSTPROCESS_MASKS", false);
  roi_mask_head.SetDatum("POSTPROCESS_MASKS_THRESHOLD", 0.5);
  roi_mask_head.SetDatum("DILATION", 1);
  //roi_mask_head.SetDatum("USE_GN", false);
  default_config.SetDatum("ROI_MASK_HEAD", roi_mask_head);

  //RESNET
  CFG resnet = CFG();
  resnet.SetDatum("NUM_GROUPS", 1);
  resnet.SetDatum("WIDTH_PER_GROUP", 64);
  resnet.SetDatum("STRIDE_IN_1X1", true);
  resnet.SetDatum("TRANS_FUNC", "BottleneckWithFixedBatchNorm");
  resnet.SetDatum("STEM_FUNC", "StemWithFixedBatchNorm");
  resnet.SetDatum("RES5_DILATION", 1);
  resnet.SetDatum("BACKBONE_OUT_CHANNELS", 256 * 4);
  resnet.SetDatum("RES2_OUT_CHANNELS", 256);
  resnet.SetDatum("STEM_OUT_CHANNELS", 64);
  resnet.SetDatum("STAGE_WITH_DCN", {false, false, false, false});
  resnet.SetDatum("DEFORMABLE_GROUPS", 1);
  default_config.SetDatum("RESNET", resnet);

  //SOLVER
  CFG solver = CFG();
  solver.SetDatum("MAX_ITER", 40000);
  solver.SetDatum("BASE_LR", 0.001);
  solver.SetDatum("BASE_LR_FACTOR", 2);
  solver.SetDatum("MOMENTUM", 0.9);
  solver.SetDatum("WEIGHT_DECAY", 0.0005);
  solver.SetDatum("WEIGHT_DECAY_BIAS", 0);
  solver.SetDatum("GAMMA", 0.1);
  solver.SetDatum("STEPS", {30000});
  solver.SetDatum("WARMUP_FACTOR", 1.0 / 3.0);
  solver.SetDatum("WARMUP_ITERS", 500);
  solver.SetDatum("WARMUP_METHOD", "linear");
  solver.SetDatum("CHECKPOINT_PERIOD", 2500);
  solver.SetDatum("IMS_PER_BATCH", 16);
  default_config.SetDatum("SOLVER", solver);

  //TEST
  CFG test = CFG();
  test.SetDatum("EXPECTED_RESULTS", {});
  test.SetDatum("EXPECTED_RESULTS_SIGMA_TOL", 4);
  test.SetDatum("IMS_PER_BATCH", 8);
  test.SetDatum("DETECTIONS_PER_IMG", 100);
  default_config.SetDatum("TEST", test);

  //MISC OPTIONS
  default_config.SetDatum("OUTPUT_DIR", ".");
  //default_config.SetDatum("PATH_CATALOG", solver);
  default_config.SetDatum("DTYPE", "float32");
}

}//mrcn
}//configs