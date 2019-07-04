import argparse
import torch

import os

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer


class wrapper(torch.nn.Module):
    def __init__(self, model):
        super(wrapper, self).__init__()
        self.model = model
    
    def forward(self, input):
        output = self.model(input)
        return tuple(output)



def output_tuple_or_tensor(model, tensor):
    output = model(tensor)
    if isinstance(output, (torch.Tensor, tuple)):
        return model
    elif isinstance(output, list):
        return wrapper(model)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection jit")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight_path",
        help="weight file path"
    )
    parser.add_argument(
        "--output_path",
        help="jit output path",
        type=str,
        default='./'
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    model = build_detection_model(cfg)
    checkpoint = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpoint.load(args.weight_path, None)
    
    #backbone save
    backbone_tensor = torch.zeros(2, 3, 320, 320)
    tmp_model = output_tuple_or_tensor(model.backbone, backbone_tensor)
    backbone_jit = torch.jit.trace(tmp_model, backbone_tensor)
    backbone_jit.save(os.path.join(args.output_path, 'backbone.pth'))

    #rpn
    #conv
    conv_in_channels = model.rpn.head.conv.weight.shape[1]
    conv_tensor = torch.zeros(2, conv_in_channels, 10, 10)
    conv_jit = torch.jit.trace(model.rpn.head.conv, conv_tensor)
    conv_jit.save(os.path.join(args.output_path, 'rpn_conv.pth'))

    #cls_logits
    logits_in_channels = model.rpn.head.cls_logits.weight.shape[1]
    logits_tensor = torch.zeros(2, logits_in_channels, 10, 10)
    logits_jit = torch.jit.trace(model.rpn.head.cls_logits, logits_tensor)
    logits_jit.save(os.path.join(args.output_path, 'rpn_logits.pth'))

    #bbox_pred
    bbox_in_channels = model.rpn.head.bbox_pred.weight.shape[1]
    bbox_tensor = torch.zeros(2, bbox_in_channels, 10, 10)
    bbox_jit = torch.jit.trace(model.rpn.head.bbox_pred, bbox_tensor)
    bbox_jit.save(os.path.join(args.output_path, 'rpn_bbox.pth'))

    #box_head
    #feature extractor
    for k, v in model.roi_heads.box.feature_extractor._modules.items():
        if 'head' in k:
            #resnet head
            extractor_in_channels = v._modules['layer4'][0]._modules['downsample'][0].weight.shape[1]
            extractor_tensor = torch.zeros(2, extractor_in_channels, 10, 10)
            extractor_jit = torch.jit.trace(v, extractor_tensor)
            extractor_jit.save(os.path.join(args.output_path, 'extractor_' + k + '.pth'))
        elif 'fc' in k:
            extractor_in_channels = v.weight.shape[1]
            extractor_tensor = torch.zeros(2, extractor_in_channels)
            extractor_jit = torch.jit.trace(v, extractor_tensor)
            extractor_jit.save(os.path.join(args.output_path, 'extractor_' + k + '.pth'))
        elif 'conv' in k:
            extractor_in_channels = v.weight.shape[1]
            extractor_tensor = torch.zeros(2, extractor_in_channels, 10, 10)
            extractor_jit = torch.jit.trace(v, extractor_tensor)
            extractor_jit.save(os.path.join(args.output_path, 'extractor_' + k + '.pth'))

    #box_head
    #predictor
    cls_score_in_channels = model.roi_heads.box.predictor.cls_score.weight.shape[1]
    cls_score_tensor = torch.zeros(2, cls_score_in_channels)
    cls_score_jit = torch.jit.trace(model.roi_heads.box.predictor.cls_score, cls_score_tensor)
    cls_score_jit.save(os.path.join(args.output_path, 'cls_score.pth'))

    bbox_pred_in_channels = model.roi_heads.box.predictor.bbox_pred.weight.shape[1]
    bbox_pred_tensor = torch.zeros(2, bbox_pred_in_channels)
    bbox_pred_jit = torch.jit.trace(model.roi_heads.box.predictor.bbox_pred, bbox_pred_tensor)
    bbox_pred_jit.save(os.path.join(args.output_path, 'bbox_pred.pth'))


if __name__ == "__main__":
    main()
    print('Complete!')