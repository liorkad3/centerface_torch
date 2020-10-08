import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class WiderDataset(Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, images_dir, mode='train', transform=None):
        super(WiderDataset, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.images_dir = images_dir
        self.transform = transform

        with open(list_file) as f:
            lines = f.readlines()
        num_lines = len(lines)
        idx = 0
        while idx < num_lines:
            f_name = lines[idx].strip()
            num_faces = int(lines[idx+1].strip())
            idx = idx + 2
            box = []
            for i in range(num_faces):
                data = lines[idx + i].strip().split(' ')
                data = list(map(int, data))
                #todo to filter by invalid or other??
                x = data[0]
                y = data[1]
                w = data[2]
                h = data[3]
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])

            if len(box) > 0:
                    self.fnames.append(f_name)
                    self.boxes.append(box)
            idx = idx + num_faces

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img = self._load_image(index)
        im_width, im_height = img.size
        boxes = self.annotransform(
            np.array(self.boxes[index], dtype='float'), im_width, im_height)
        if self.transform:
            img = self.transform(img)

        return img, boxes

    # todo change to numpy array?
    def _load_image(self, index):
        image_path = f'{self.images_dir}/{self.fnames[index]}'
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        return img
        
        
    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


if __name__ == '__main__':
    # from config import cfg
    dataset = WiderDataset('data/wider_face_split/wider_face_val_bbx_gt.txt',
    images_dir='data/WIDER_VAL/images', mode='val')
    #for i in range(len(dataset)):
    img, boxes = dataset[3]
    print()
