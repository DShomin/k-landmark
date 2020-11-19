import os
import time
import numpy as np
import torch
import torch.nn as nn
from evaluations.kaggle_2020.global_average_precision  import global_average_precision_score
from torch.cuda import amp

def train(args, trn_cfg):
    
    train_loader = trn_cfg['train_loader']
    valid_loader = trn_cfg['valid_loader']
    model = trn_cfg['model']
    criterion = trn_cfg['criterion']
    optimizer = trn_cfg['optimizer']
    scheduler = trn_cfg['scheduler']
    device = trn_cfg['device']

    best_epoch = 0
    best_val_score = 0.0

    print('Training Start...')
    if args.Warmup:
        optimizer.zero_grad()
        optimizer.step()
    # scaler = amp.GradScaler()
    # Train the model
    for epoch in range(args.epochs):
        
        start_time = time.time()
    
        # train
        model.train()
        trn_loss = 0.0
        if args.Warmup:
            scheduler.step(epoch)

        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            if device:
                images = images.to(device)
                labels = labels.to(device) # 
            # print(labels.shape)
            
            # Forward pass
            # with amp.autocast():

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()

        epoch_train_loss = trn_loss / len(train_loader)


        # validation
        # val_loss, val_score = validation(args, trn_cfg, model, criterion, valid_loader, device)
        model.eval()
        val_loss = 0.0
        total_labels = []
        total_outputs = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_loader):
                
                total_labels.append(labels)

                if device:
                    images = images.to(device)
                    labels = labels.to(device) #.reshape(-1, 1)

                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                total_outputs.append(torch.softmax(outputs.cpu().detach(), dim=1).numpy())

        epoch_val_loss = val_loss / len(valid_loader)
        
        total_labels = np.concatenate(total_labels).tolist()
        total_outputs = np.concatenate(total_outputs)

        landmark_id = np.argmax(total_outputs, axis=1).tolist()
        conf = np.max(total_outputs, axis=1).tolist()
        pred = {str(i) : (pred_id, pred_conf) for i, (pred_id, pred_conf) in enumerate(zip(landmark_id, conf))}
        label = {str(i) : label_id for i, label_id in enumerate(total_labels)}
        val_score = global_average_precision_score(label, pred)

        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - trn_loss: {:.4f}  val_loss: {:.4f}  val_score: {:.4f} lr: {:.5f}  time: {:.0f}s\n".format(
                epoch+1, epoch_train_loss, epoch_val_loss, val_score, lr[0], elapsed))
        if not args.Warmup:
            scheduler.step(val_score)
        # scheduler.step()

        
        # save model weight
        if val_score > best_val_score:
            best_val_score = val_score            
            file_save_name = 'best_score_ep30_sz448_rer_lrup_b1' + '_fold' + str(args.fold_num) + '.pt'

            torch.save(model.state_dict(), os.path.join('../models', file_save_name))

            