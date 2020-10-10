from preprocessing.transforms import ConvertFromInts, ToTensor, Resize, ToPercentCoords, Padding
from preprocessing.label_transforms import ToHeatmaps, MaxObjects

class TrainAugmentation:
    def __init__(self, size, stride=4, max_obj=32):
        self.size = size
        self.stride = stride
        self.max_obj = max_obj
        self.augment = Compose([
            ConvertFromInts(),
            # ToPercentCoords(),
            Resize(self.size),
            ToTensor(),
            Padding(self.size)
        ])
        self.label_transforms = ComposeLabel([
            MaxObjects(self.max_obj),
            ToHeatmaps(self.size, self.stride, self.max_obj)
        ])

    def __call__(self, img, boxes):
        img, boxes = self.augment(img, boxes)
        batch = self.label_transforms(boxes)
        batch['input'] = img
        return batch

class EvalAugmentation:
    def __init__(self, size):
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            # ToPercentCoords(),
            Resize(self.size),
            ToTensor(),
            Padding(self.size)
        ])

    def __call__(self, img, boxes):
        return self.augment(img, boxes)

class Compose:
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes

class ComposeLabel:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, boxes):
        for t in self.transforms:
            boxes = t(boxes)
        return boxes

if __name__ == "__main__":
    trans = TrainAugmentation(640, 32)
    import cv2
    import numpy as np
    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = np.array(([120, 150, 500, 400], [0, 38, 250, 600]))
    img, boxes, size = trans(img, boxes)
    print()