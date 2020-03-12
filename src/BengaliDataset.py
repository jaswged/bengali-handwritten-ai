import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


# DataSet creator
class BengaliDataset(Dataset):
    def __init__(self, df, img_height, img_width):
        self.df = df
        self.img_height = img_height
        self.img_width = img_width

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.df.iloc[idx][0:].values.astype(np.uint8)
        img = img.reshape(self.img_height, self.img_width)
        img = 255 - img
        img = self.crop_resize(img)
        img = img[:, :, np.newaxis]

        return (self.df.index[idx],
                torch.tensor(img, dtype=torch.float).permute(2, 0, 1))

    # Image resizing and reshaping
    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def crop_resize(self, img0, size=SIZE, pad=16):
        # crop a box around pixels large than the threshold
        # some images contain line at the sides
        ymin, ymax, xmin, xmax = self.bbox(img0[5:-5, 5:-5] > 60)
        # cropping may cut too much, so we need to add it back
        xmin = xmin - 13 if (xmin > 13) else 0
        ymin = ymin - 10 if (ymin > 10) else 0
        xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
        ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
        img = img0[ymin:ymax, xmin:xmax]
        # remove lo intensity pixels as noise
        img[img < 28] = 0
        lx, ly = xmax - xmin, ymax - ymin
        l = max(lx, ly) + pad
        # make sure that the aspect ratio is kept in rescaling
        img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)],
                     mode='constant')
        return cv2.resize(img, (size, size))
