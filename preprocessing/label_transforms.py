from preprocessing.image_utils import draw_msra_gaussian, gaussian_radius, draw_umich_gaussian
import numpy as np
import math

class ToHeatmaps:
    def __init__(self, size, stride, max_objs):
        self.size = size // stride
        self.stride = stride
        self.max_objs = max_objs
    def __call__(self, boxes):
        hm = np.zeros((self.size, self.size), dtype=np.float32) #hm for heatmap
        scales = np.zeros((self.max_objs, 2), dtype=np.float32)      #s for scales
        ind = np.zeros((self.max_objs), dtype=np.int64)         #index in output of gt_boxes
        mask = np.zeros((self.max_objs), dtype=np.int64)
        off = np.zeros((self.max_objs, 2), dtype=np.float32)    #off for offsets

        draw_gaussian = draw_umich_gaussian                     # for focal loss
        for k, box in enumerate(boxes):
            h, w = box[3] - box[1], box[2] - box[0]

            ct = np.array(
            [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)
            ct = ct / self.stride
            ct_int = ct.astype(np.int32)

            scales[k] = np.log(1. * w / self.stride), np.log(1. * h / self.stride) 
            off[k] = ct - ct_int 
            ind[k] = ct_int[1] * self.size + ct_int[0] 
            mask[k] = 1  

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(int(radius), 0)
            draw_gaussian(hm, ct_int, radius)
        hm = np.expand_dims(hm, 0)
        ret = {'hm':hm, 'scale':scales, 'off':off, 'ind':ind, 'mask':mask}
        return ret

class MaxObjects:
    def __init__(self, max_obj):
        self.max_obj = max_obj
    def __call__(self, boxes):
        boxes = np.array(boxes)
        if len(boxes) > self.max_obj:    # choose max objects randomaly
            boxes = boxes[np.random.choice(boxes.shape[0],  self.max_obj), :]
        return boxes


if __name__ == "__main__":
    t_hm = ToHeatmaps(32, 32)
    m = MaxObjects(32)
    box1 = np.array([1, 1, 10, 10])
    box2 = np.array([8, 8, 28, 28])
    boxes = [box1, box2]
    boxes = boxes *20
    boxes = m(boxes)
    ret = t_hm(boxes)
    print()