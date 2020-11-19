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

        # self.img_list = list()

        # for img in self.images:
        #     loaded_img = cv2.imread(img + '.JPG')
        #     loaded_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)
        #     loaded_img = Image.fromarray(loaded_img)
        #     self.img_list.append(loaded_img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.base_path, self.images[idx])
        img_path = self.images[idx]
        x = cv2.imread(img_path+'.JPG')
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = Image.fromarray(x)
        # x = self.img_list[idx]
        if self.default_transforms is not None:
            x = self.default_transforms(x)
        if self.trans is not None:
            x = self.trans(x)

        if not self.is_test:
            y = self.labels[idx]
            return (torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
        else:
            return torch.as_tensor(x, dtype=torch.float32)


def build_dataloder(df, CURRENT_FOLD, BASE_PATH, default_transforms, train_transform, valid_transform, batch_size, num_workers, pseudo_label):

    df.id = df.id.apply(lambda x : os.path.join(BASE_PATH, x))

    train_df = df.loc[df.fold != CURRENT_FOLD, ['id', 'landmark_id']]
    # train_df = df # all dataset
    valid_df = df.loc[df.fold == CURRENT_FOLD, ['id', 'landmark_id']]

    # load test dataset
    if pseudo_label:
        pred_df = pd.read_csv('../sub_ep30_fold0.csv')
        pred_df = pred_df.loc[pred_df.conf >= 0.003, ['id', 'landmark_id']]
        pred_df.id = pred_df.id.apply(lambda x : os.path.join('../data/public/test', x))
        train_df = train_df.append(pred_df)

    trn_dataset = landmark_dataset(train_df, train_transform, is_test=False, BASE_PATH=BASE_PATH, default_transforms=default_transforms)
    val_dataset = landmark_dataset(valid_df, valid_transform, is_test=False, BASE_PATH=BASE_PATH, default_transforms=default_transforms)

    trn_dataloder = DataLoader(trn_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    val_dataloder = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    return trn_dataloder, val_dataloder
