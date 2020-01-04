import torch
import torch.nn as nn

from typing import List

import numpy as np


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> List[float]:
    """
    Calculate intersection over union of detection of interest and all other detections. Assumes that boxes have been
    put in corner coordinate format.

    :param box1: Detection of interest.
    :param box2: Other detections.
    :return: List of IOU values.
    """
    box2 = box2.t()
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # intersection
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # union
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area


def center_to_corners(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2] (i.e. from use of center coordinate to corner
    coordinates). This makes IOU calculation easier.

    :param x: Predictions in [x, y, w, h] format.
    :return: Predictions in [x1, y1, x2, y2] format.
    """
    box_coords = torch.zeros_like(x)
    box_coords[:, 0] = x[:, 0] - x[:, 2] / 2
    box_coords[:, 1] = x[:, 1] - x[:, 3] / 2
    box_coords[:, 2] = x[:, 0] + x[:, 2] / 2
    box_coords[:, 3] = x[:, 1] + x[:, 3] / 2
    return box_coords

def non_max_suppression(predictions: List[torch.Tensor], conf_thresh: float = 0.5, nms_thresh: float = 0.5) -> List[torch.Tensor]:
    """
    Performs non max suppression.
    : Choose box, X, from predictions with highest confidence score. Compare it to other boxes. If a box, Y, being
    : compared has a higher IOU with X than the nms_thresh, it can be assumed that Y is trying to predict the same
    : object. So, Y can be removed from the pool of predictions. Repeat this until there are no boxes left in the pool
    : of predictions.

    :param predictions: Output from the network containing all predictions.
    :param conf_thresh: Threshold for confidence values, over which we can say a prediction is valid.
    :param nms_thresh: Threshold against which to check IOUs.
    :return: Narrowed pool of predictions. Shape = (x1, y1, x2, y2, object_conf, class_conf, class).
    """
    output = []
    min_pixels = 2      # minimum pixel width allowed for predictions

    for pred in predictions:
        pred = pred.squeeze()
        # multiply objectness by class conf to get combined confidence
        # i.e. how sure object is boxed combined with how sure object is given class
        class_conf, class_pred = pred[:, 5:].max(1)
        pred[:, 4] *= class_conf

        # select only suitable predictions
        # i.e. object with confidence above threshold, more than minimum pixel requirement, and with finite elements
        good_preds = (pred[:, 4] > conf_thresh) & (pred[:, 2:4] > min_pixels).all(1) & torch.isfinite(pred).all(1)
        pred = pred[good_preds]

        # no need to continue if no predictions deemed good
        if len(pred) == 0:
            continue

        # select predicted classes
        class_conf = class_conf[good_preds]
        class_pred = class_pred[good_preds].unsqueeze(1).float()

        # change coordinates from center to corners
        pred[:, :4] = center_to_corners(pred[:, :4])

        # format detections as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred = torch.cat((pred[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # sort detections by decreasing confidence scores
        pred = pred[(-pred[:, 4]).argsort()]

        det_max = []

        for cl in pred[:, -1].unique():
            det_cl = pred[pred[:, -1] == cl]  # select detected instances (predictions) of class cl
            n = len(det_cl)
            if n == 1:
                det_max.append(det_cl)  # no NMS required if only 1 prediction
                continue
            elif n > 100:
                # limit to first 100 boxes to speed training
                # NMS is O(N^2) in the number of boxes, because you potentially check each box against each other box.
                # With thousands of candidate boxes this process is very slow.
                det_cl = det_cl[:100]

            while det_cl.shape[0]:
                det_max.append(det_cl[:1])  # save highest confidence detection
                if len(det_cl) == 1:  # stop if we're at the last detection
                    break
                iou = calculate_iou(det_cl[0], det_cl[1:])  # iou with other boxes
                det_cl = det_cl[1:][iou < nms_thresh]  # remove ious > threshold

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output.append(det_max[(-det_max[:, 4]).argsort()])  # sort

    return output
