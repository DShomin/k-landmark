from create_dataloader import landmark_dataset
from transforms import get_transform
from utils import seed_everything
from loss_function import AngularPenaltySMLoss
from create_model import build_model
from torch.utils.data import DataLoader
from scipy.special import softmax

import pandas as pd
import numpy as np
import torch
import os

BASE_PATH = '../data/public/test'

MODEL_LIST = [
    '../models/best_score_ep30_gap_b0_fold0.pt',
    # '../models/best_score_320_ep50_napm_fold0.pt'
                # '../models/best_score_320_stplr_napm_fold0.pt',
                # '../models/best_score_320_sch_napm_fold0.pt',
                # '../models/best_score_320_fold0.pt',
                # '../models/best_score_fold0.pt',
                # '../models/best_score_fold1.pt',
                #  '../models/best_score_fold2.pt'
                 ]
IMG_SIZE = (320, 320)
TTA = 1
seed_everything(42)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('../data/public/sample_submisstion.csv')
df = df#[:20]

test_trans = get_transform(IMG_SIZE, transform_list='') #horizontal_flip

test_dataset = landmark_dataset(df, test_trans, is_test=True, BASE_PATH=BASE_PATH)
test_dataloader = DataLoader(test_dataset,  batch_size=32, num_workers=6, pin_memory=True, shuffle=False)

# define model
model = build_model('efficientnet-b0', do_gem=True)
model = model.to(device)

model_pred = list()
for MODEL_PATH in MODEL_LIST:
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()
    for _ in range(TTA):
        total_outputs = []
        with torch.no_grad():
            for batch_idx, (images) in enumerate(test_dataloader):

                if device:
                    images = images.to(device)

                outputs = model(images)
                total_outputs.append(torch.softmax(outputs.cpu().detach(), dim=1).numpy())

        total_outputs = np.concatenate(total_outputs)
        model_pred.append(np.expand_dims(total_outputs, axis=0))

model_pred = np.concatenate(model_pred)
model_pred = model_pred.mean(axis=0)

landmark_id = np.argmax(total_outputs, axis=1)#.tolist()
conf = np.max(total_outputs, axis=1)#.tolist()

df['landmark_id'] = landmark_id
df['conf'] = conf + 0.0001

print(df.head())

df.to_csv('../sub_ep30_gap_fold0.csv', index=False)