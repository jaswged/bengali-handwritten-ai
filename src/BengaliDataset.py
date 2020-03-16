import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm


class BengaliDataset(Dataset):
    def __init__(self, df, img_height, img_width):
        self.df = df
        self.img_height = img_height
        self.img_width = img_width
        self.HEIGHT = 137
        self.WIDTH = 236

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.df.iloc[idx][0:].values.astype(np.uint8)
        img = img.reshape(self.img_height, self.img_width)

        # Invert the colors of the image.
        img = 255 - img
        # Crop and resize the image
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

    def crop_resize(self, img0, size=128, pad=16, threshold=70):
        # crop a box around pixels large than the threshold
        # some images contain line at the sides
        ymin, ymax, xmin, xmax = self.bbox(img0[5:-5, 5:-5] > threshold)
        # cropping may cut too much, so we need to add it back
        xmin = xmin - 13 if (xmin > 13) else 0
        ymin = ymin - 10 if (ymin > 10) else 0
        xmax = xmax + 13 if (xmax < self.WIDTH - 13) else self.WIDTH
        ymax = ymax + 10 if (ymax < self.HEIGHT - 10) else self.HEIGHT
        img = img0[ymin:ymax, xmin:xmax]

        # remove lo intensity pixels as noise. 28 is cutoff
        img[img < 28] = 0
        length_x, length_y = xmax - xmin, ymax - ymin
        length = max(length_x, length_y) + pad

        # make sure that the aspect ratio is kept in rescaling
        img = np.pad(img, [((length - length_y) // 2,),
                           ((length - length_x) // 2,)], mode='constant')
        return cv2.resize(img, (size, size))

    def save_images(self, output_file):
        x_tot, x2_tot = [], []
        with zipfile.ZipFile(output_file, 'w') as img_out:
            # the input is inverted
            data = 255 - self.df.iloc[:, 1:]\
                .values.reshape(-1, self.HEIGHT, self.WIDTH)\
                .astype(np.uint8)

            for idx in tqdm(range(len(self.df))):
                name = self.df.iloc[idx, 0]
                # normalize each image by its max val
                img = (data[idx] * (
                            255.0 / data[idx].max())).astype(
                    np.uint8)
                img = self.crop_resize(img)

                x_tot.append((img / 255.0).mean())
                x2_tot.append(((img / 255.0) ** 2).mean())
                img = cv2.imencode('.png', img)[1]
                img_out.writestr(name + '.png', img)


class BenDataset(Dataset):
    def __init__(self, img_height, img_width):
        self.dataset = []
        self.img_height = img_height
        self.img_width = img_width
        self.HEIGHT = 137
        self.WIDTH = 236

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def add_BengaliDataset(self, bdg:BengaliDataset):
        for i in range(len(bdg)):
            self.dataset.append(bdg.__getitem__(i))
