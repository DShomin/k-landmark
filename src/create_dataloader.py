from torch.utils.data import DataLoader, Dataset
from glob import glob
from PIL import Image
import pandas as pd
import numpy as np
import os
import cv2
import torch

class landmark_dataset(Dataset):
    def __init__(self, df, trans, is_test, BASE_PATH, default_transforms=None):
        self.images = df.id.values
        self.labels = df.landmark_id.values
        self.trans = trans

        self.base_path = BASE_PATH
        self.is_test = is_test
        self.default_transforms = default_transforms

        # self.imgs = []
        
        # for img_path in glob(BASE_PATH + '*.JPG'):
        #     img = cv2.imread(img_path)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     img = Image.fromarray(img)

        #     if default_transforms is not None:
        #         img = default_transforms(img)
    
        #     self.imgs.append(img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.images[idx])
        x = cv2.imread(img_path+'.JPG')
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = Image.fromarray(x)

        if self.trans is not None:
            if self.default_transforms is not None:
                x = self.default_transforms(x)
            x = self.trans(x)

        if not self.is_test:
            y = self.labels[idx]
            return (torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
        else:
            return torch.as_tensor(x, dtype=torch.float32)


def build_dataloder(df, CURRENT_FOLD, BASE_PATH, default_transforms, train_transform, valid_transform, batch_size, num_workers):

    train_df = df.loc[df.fold != CURRENT_FOLD]
    valid_df = df.loc[df.fold == CURRENT_FOLD]

    trn_dataset = landmark_dataset(train_df, train_transform, is_test=False, BASE_PATH=BASE_PATH, default_transforms=default_transforms)
    val_dataset = landmark_dataset(valid_df, valid_transform, is_test=False, BASE_PATH=BASE_PATH, default_transforms=default_transforms)

    trn_dataloder = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    val_dataloder = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    return trn_dataloder, val_dataloder
