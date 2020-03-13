import numpy as np

# Pytorch
import pandas as pd
import torch
from Utils import seed_everything
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F

# utils
import random
import os
import gc
from tqdm import tqdm

from BengaliDataset import BengaliDataset
from Resnet import ResNet


# Constants
# Setting
SEED = 222
BATCH_SIZE = 64
HEIGHT = 137
WIDTH = 236
device = "cuda" if torch.cuda.is_available() else "cpu"

# Call seed to setup for repeatable results
seed_everything(SEED)

# Create Model
model = ResNet().to(device)

# Setup data files to load
data_dir = '../input/bengaliai-cv19'
files_train = [f'train_image_data_{fid}.parquet' for fid in range(4)]

# Predict Test values
model.eval()
row_id = []
target = []

# Do training here
for fname in files_train:
    # Read in the data files
    file = os.path.join(data_dir, fname)
    print(F"File name is {file}")
    df_test = pd.read_parquet(file)
    print(F"Shape of this parquet dataframe is: {df_test.shape}")
    df_test.set_index('image_id', inplace=True)

    test_image = BengaliDataset(df_test, img_height=HEIGHT,
                                img_width=WIDTH)
    test_loader = torch.utils.data.DataLoader(dataset=test_image,
                                              batch_size=BATCH_SIZE,
                                              num_workers=4,
                                              shuffle=False)

    with torch.no_grad():
        for idx, (img_ids, img) in enumerate(test_loader):
            img = img.to(device)
            pred_graphemes, pred_vowels, pred_consonants = model(img)
            for img_id, pred_grapheme, pred_vowel, pred_consonant in zip(
                    img_ids, pred_graphemes, pred_vowels,
                    pred_consonants):
                row_id.append(img_id + '_consonant_diacritic')
                target.append(
                    pred_consonant.argmax(0).cpu().detach().numpy())
                row_id.append(img_id + '_grapheme_root')
                target.append(
                    pred_grapheme.argmax(0).cpu().detach().numpy())
                row_id.append(img_id + '_vowel_diacritic')
                target.append(
                    pred_vowel.argmax(0).cpu().detach().numpy())

    del (df_test, test_image, test_loader)
    gc.collect()
