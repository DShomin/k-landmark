from transforms import get_transform
from make_fold import make_fold
from create_dataloader import build_dataloder
from create_model import build_model
from optimizer import build_optimizer, build_scheduler
from loss_function import AngularPenaltySMLoss
from trainer import train
from utils import seed_everything
from torch import nn
from torchvision import transforms

import pandas as pd
import torch
import argparse
import gc

TRAIN_PATH = '../data/public/train/'
CURRENT_FOLD = 0
IMG_SIZE = (224, 448)
BATCH_SIZE = 8
NUM_WORKERS = 4
SEED = 42


def main(args):

    # fix seed for train reproduction
    seed_everything(SEED)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device", device)

    df = pd.read_csv('../data/public/train.csv')
    df = make_fold(df, how='stratified')

    ############ DONOTCHANGE: Path loader ###############
    if args.DEBUG: 
        total_num = 100
        df = df[:total_num]

    # default_transforms = transforms.Compose([transforms.Resize(IMG_SIZE)])
    default_transforms = None


    trn_trans = get_transform(target_size=IMG_SIZE,
                                        transform_list=args.train_augments, 
                                        augment_ratio=args.augment_ratio)
    val_trans = get_transform(target_size=IMG_SIZE,
                                        transform_list=args.valid_augments, 
                                        augment_ratio=args.augment_ratio,
                                        is_train=False)

    trn_dataloader, val_dataloader = build_dataloder(df, CURRENT_FOLD, TRAIN_PATH, default_transforms, trn_trans, val_trans, BATCH_SIZE, NUM_WORKERS)


    # define model
    model = build_model(args.model, do_gem=True)
    model = model.to(device)
    
    # optimizer definition
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, len(trn_dataloader))

    criterion = AngularPenaltySMLoss(1049, loss_type='cosface').to(device) # loss_type in ['arcface', 'sphereface', 'cosface']
    # criterion = nn.CrossEntropyLoss() # BCELoss()

    trn_cfg = {'train_loader':trn_dataloader,
                'valid_loader':val_dataloader,
                'model':model,
                'criterion':criterion,
                'optimizer':optimizer,
                'scheduler':scheduler,
                'device':device,
                }

    train(args, trn_cfg)

    del model, trn_dataloader, val_dataloader
    gc.collect()

if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    # custom args
    arg('--epochs', type=int, default=20, help='number of epochs to train')
    arg('--model', type=str, default='efficientnet-b0')
    arg('--optimizer', type=str, default='SGD')
    arg('--scheduler', type=str, default='Plateau', help='scheduler in steplr, plateau, cosine')
    arg('--lr', type=float, default=1e-4) 
    arg('--weight_decay', type=float, default=0.0) 
    arg('--train_augments', type=str, default='horizontal_flip')
    arg('--valid_augments', type=str, default='horizontal_flip')
    arg('--augment_ratio', default=0.5, type=float, help='probability of implementing transforms')
    arg('--lookahead', default=False, type=bool, help='use lookahead')
    arg('--k_param', type=int, default=5)
    arg('--alpha_param', type=float, default=0.5)
    arg('--patience', type=int, default=3, help='plateau scheduler patience parameter')
    arg('--DEBUG', default=False, type=bool, help='if true debugging mode')
    args = parser.parse_args()
    args.fold_num = CURRENT_FOLD

    main(args)