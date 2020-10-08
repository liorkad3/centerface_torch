import cv2 as cv2
import numpy as np
import torch

class ConvertFromInts:
    def __call__(self, image, boxes=None):
        return image.astype(np.float32), boxes.astype(np.float32)
class ToTensor:
    def __call__(self, cvimage, boxes=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes

class Resize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img, boxes=None):
        h, w, _ = img.shape
        max_size = max(h, w)
        scale = self.size / max_size
        h_new, w_new = int(scale * h), int(scale * w)
        img = cv2.resize(img, (w_new, h_new))
        return img, boxes

class ToPercentCoords:
    def __call__(self, image, boxes=None):
        height, width, _ = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes