import torch
import numpy as np


class Cutout(object):
    """
    Taken primarily from: https://github.com/uoguelph-mlrg/Cutout
    Randomly mask out one or more patches from an image.
    """
    def __init__(self, n_holes, length):
        """
        :param n_holes: Number of patches to cut out of each image.
        :param length: The length (in pixels) of each square patch.
        """
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        :param img: Tensor image of size (C, H, W).
        :return: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
