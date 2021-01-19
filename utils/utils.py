import numpy as np
import cv2
from PIL import Image

def compute_overlap(a, b):

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    inter_w = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0])
    inter_h = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1])

    inter_w = np.maximum(0, inter_w)
    inter_h = np.maximum(0, inter_h)

    intersect =  inter_w * inter_h
    union = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - intersect
    union = np.maximum(union, np.finfo(float).eps)
    return intersect / union


def compute_ap(recall, precision):

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap