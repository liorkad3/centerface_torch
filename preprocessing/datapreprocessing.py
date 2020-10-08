from transforms import ConvertFromInts, ToTensor, Resize, ToPercentCoords

class TrainAugmentation:
    def __init__(self, size):
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(self.size),
            ToTensor()
        ])

    def __call__(self, img, boxes):
        return self.augment(img, boxes)

class EvalAugmentation:
    def __init__(self, size):
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToPercentCoords(),
            Resize(self.size),
            ToTensor()
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

if __name__ == "__main__":
    trans = TrainAugmentation(640)
    import cv2
    import numpy as np
    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = np.array(([120, 150, 500, 400], [0, 38, 250, 600]))
    img, boxes = trans(img, boxes)