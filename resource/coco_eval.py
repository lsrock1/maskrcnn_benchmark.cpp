import os
from collections import OrderedDict
import sys


def do_coco_evaluation(
    coco_path,
    output_folder,
    iou_types
):

    coco_results = {}
    # if "bbox" in iou_types:
    #     logger.info("Preparing bbox results")
    #     coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
    # if "segm" in iou_types:
    #     logger.info("Preparing segm results")
    #     coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)

    results = COCOResults(*iou_types)

    for iou_type in iou_types:
        file_path = os.path.join(output_folder, iou_type + ".json")
        res = evaluate_predictions_on_coco(
            coco_path, file_path, iou_type
        )
        results.update(res)


def evaluate_predictions_on_coco(
    coco_path, json_result_file, iou_type="bbox"
):

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(coco_path)
    coco_dt = coco_gt.loadRes(str(json_result_file)) if json_result_file else COCO()

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results


# def check_expected_results(results, expected_results, sigma_tol):
#     if not expected_results:
#         return

#     logger = logging.getLogger("maskrcnn_benchmark.inference")
#     for task, metric, (mean, std) in expected_results:
#         actual_val = results.results[task][metric]
#         lo = mean - sigma_tol * std
#         hi = mean + sigma_tol * std
#         ok = (lo < actual_val) and (actual_val < hi)
#         msg = (
#             "{} > {} sanity check (actual vs. expected): "
#             "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
#         ).format(task, metric, actual_val, mean, std, lo, hi)
#         if not ok:
#             msg = "FAIL: " + msg
#             logger.error(msg)
#         else:
#             msg = "PASS: " + msg
#             logger.info(msg)

if __name__ == '__main__':
    coco_path = sys.argv[1]
    output_folder = sys.argv[2]
    iou_types = sys.argv[3:]
    do_coco_evaluation(coco_path, output_folder, iou_types)