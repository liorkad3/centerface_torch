import cv2 as cv2
import numpy as np
import torch
import torch.nn.functional as F
from math import ceil, floor

class ConvertFromInts:
    def __call__(self, image, boxes=None):
        return image.astype(np.float32), boxes.astype(np.float32)
class ToTensor:
    def __call__(self, cvimage, boxes=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes

class Padding:
    def __init__(self, size):
        self.size = size
    def __call__(self, in_tensor, boxes=None):
        _, h, w= in_tensor.shape
        w_pad = (self.size - w) / 2
        h_pad = (self.size - h) / 2
        pad_mat = (ceil(w_pad), floor(w_pad), ceil(h_pad), floor(h_pad))
        in_tensor = F.pad(in_tensor, pad_mat, mode='constant', value=0)
        boxes[:, 0] += pad_mat[0]
        boxes[:, 2] += pad_mat[0]
        boxes[:, 1] += pad_mat[2]
        boxes[:, 3] += pad_mat[2]
        return in_tensor, boxes

class Resize:
    def __init__(self, size):
        self.size = size
    def __call__(self, img, boxes=None):
        h, w, _ = img.shape
        max_size = max(h, w)
        scale = self.size / max_size
        h_new, w_new = int(scale * h), int(scale * w)
        img = cv2.resize(img, (w_new, h_new))
        boxes = boxes * scale
        return img, boxes

class ToPercentCoords:
    def __call__(self, image, boxes=None):
        height, width, _ = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes
    


if __name__ == "__main__":
    # p = Padding(640)
    # x = torch.randn(3, 640, 603)
    # y = p(x)
    # print(y.shape)

    import cv2
    import numpy as np
    img = cv2.imread('test2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = np.array(([120, 150, 500, 400], [0, 38, 250, 600]))
    r = Resize(640)
    p = Padding(640)
    t = ToTensor()
    print(boxes)
    i, b = r(img, boxes)
    print(b)
    i, _ = t(i)
    i, b2 = p(i, b)
    print(b2)
    print(i.shape)