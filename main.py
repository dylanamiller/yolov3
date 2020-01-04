import torch
import cv2
import numpy as np
import argparse

from darknet import Darknet
from detect import *


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    img_ = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = img_.clone() # torch.tensor(img_)  # Convert to Variable
    return img_


def main(num_blocks, anchors, routes, conf_thresh, nms_thresh):
    model = Darknet(num_blocks, anchors, routes)

    img = get_test_input()
    pred = model(img)
    pred = non_max_suppression(pred, conf_thresh, nms_thresh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--nms_thresh', type=float, default=0.5)
    parser.add_argument('--num_blocks', nargs='*', default=[1, 2, 8, 8, 4])
    parser.add_argument('--anchors', nargs='*',
                        default=[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                 (59, 119), (116, 90), (156, 198), (373, 326)])
    parser.add_argument('--routes', nargs='*', default=[(-4, -1, 61), (-4, -1, 36)])
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.set_defaults(train=False, gpu=False)

    args = parser.parse_args()

    if args.gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)

    main(args.num_blocks, args.anchors, args.routes, args.conf_thresh, args.nms_thresh)
