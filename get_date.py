import collections.abc
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Tuple
import cv2
import numpy as np
from megengine.data.dataset.vision.meta_vision import VisionDataset

class u_data(VisionDataset):
    supported_order = ("image", "mask",)

    def __init__(self, root, *, order=None):
        super().__init__(root, order=order)

        voc_root = self.root
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted.")

        image_dir = os.path.join(voc_root, "image")
        mask_dir = os.path.join(voc_root, "label")
        self.file_names = os.listdir(image_dir)

        self.images = [os.path.join(image_dir, x) for x in self.file_names]
        self.masks = [os.path.join(mask_dir, x) for x in self.file_names]

    def __getitem__(self, index):
        target = []
        for k in self.order:
            if k == "image":
                image = cv2.imread(self.images[index], cv2.IMREAD_GRAYSCALE)
                image = image / 255.
                image = image[:, :, np.newaxis].astype(np.float32)
                target.append(image)

            elif k == "mask":
                mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i][j] == 0:
                            mask[i][j] = 0
                        elif mask[i][j] == 80:
                            mask[i][j] = 1
                        elif mask[i][j] == 160:
                            mask[i][j] = 2
                        elif mask[i][j] == 255:
                            mask[i][j] = 3
                        else:
                            mask[i][j] = 0

                mask = mask[:, :, np.newaxis].astype(np.int32)
                target.append(mask)
            else:
                raise NotImplementedError

        return tuple(target)

    def __len__(self):
        return len(self.images)
