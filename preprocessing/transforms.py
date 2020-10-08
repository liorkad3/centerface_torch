import cv2 as cv2
import numpy as np
import torch

class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)
class ToTensor(object):
    def __call__(self, cvimage):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)