import torch
import cv2 as cv2
from torch.utils.data import Dataset
import numpy as np


class WiderDataset(Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, data_dir, mode='train', transform=None):
        super(WiderDataset, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.transform = transform

        mode = 'val'
        if mode == 'train':
            self.images_dir = f'{data_dir}/WIDER_TRAIN/images'
            self.list_file = f'{data_dir}/wider_face_split/wider_face_train_bbx_gt.txt'
        elif mode == 'val':
            self.images_dir = f'{data_dir}/WIDER_VAL/images'
            self.list_file = f'{data_dir}/wider_face_split/wider_face_val_bbx_gt.txt'
        else:
            raise NameError(f'not supported mode: {mode}')

        with open(self.list_file) as f:
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
        boxes = np.array(self.boxes[index], dtype='float')
        if self.transform:
            batch = self.transform(img, boxes)
        return batch

    def _load_image(self, index):
        image_path = f'{self.images_dir}/{self.fnames[index]}'
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
        

if __name__ == '__main__':
    trans = TrainAugmentation(640)
    dataset = WiderDataset(data_dir='data', mode='val', transform=trans)
    img, boxes = dataset[3]
    print(img.shape, boxes[0])
