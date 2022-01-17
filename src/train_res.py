
import time

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score

from models import resnet18, resnet18_transforms, ResNet18
from utils import PatchDataset, Tensorboard

if __name__ == '__main__':
    # Configuration parameters.
    SCALE = 0.9
    PAIRS = 4
    DATASET = 'Celeb-DF-v2'
    PATCHES_DIR = f'/hdd2/vol1/deepfakeDatabases/anzem-cropped_videos/patches/{DATASET}/{SCALE}/{PAIRS}'

    BATCH_SIZE = 128
    LR = 0.001

    optimizer_params = {
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': True
    }

    lr_scheduler_params = {
        'step_size': 2,
        'gamma': 0.5,
        'verbose': True
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    X = PatchDataset(f'{PATCHES_DIR}/train', PAIRS, neg_split=[0, 19000], pos_split=[20000, 101000])
    class_weights = X.get_balanced_classes_weights()
    train_dl = DataLoader(
        X,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=4,
        sampler=WeightedRandomSampler(class_weights, len(class_weights), replacement=True),
        # shuffle=True
    )

    X_val = PatchDataset(f'{PATCHES_DIR}/validation', PAIRS, neg_split=[0, 2000], pos_split=[5000, 7000])
    val_dl = DataLoader(
        X_val,
        batch_size=BATCH_SIZE,
        drop_last=True,
        num_workers=4,
        # shuffle=True
    )

    model = resnet18(pretrained=True).to(device)
    # print(ResNet18.__mro__)

    # Freeze all of the layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the fc layers
    model.fc_face.weight.requires_grad = True
    model.fc_face.bias.requires_grad = True

    model.fc_pair.weight.requires_grad = True
    model.fc_pair.bias.requires_grad = True


    model = nn.DataParallel(model.cuda())

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn.cuda()

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=0.001) #0.05, try 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001) # Not pretrained configs.
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, verbose=True) # Try: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html

    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.25, verbose=True) # try 0.75
    train_writer = Tensorboard('./logs/logs_train')
    val_writer = Tensorboard('./logs/logs_val')
    acc_writer = Tensorboard('./logs/logs_acc')

    model_name = f'./trained_models/resnet18-{time.time()}'

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    EPOCH = 10
    for epoch in range(EPOCH):
        avg_loss = []
        avg_val_loss = []
        epoch_acc = []
 
        for i, (batch_X, batch_y) in enumerate(tqdm(train_dl)):
            # Training
            model.train()

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = []
            for sample in range(BATCH_SIZE):
                out = model(batch_X[sample])
                outputs.append(out)

            loss = loss_fn(torch.stack(outputs).to(device), batch_y)
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            
            train_writer.plot_loss(loss, i, loss_name=f'loss_epoch_{epoch}')
            

            # Validation
            # if i % 10 == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         for batch_X_val, batch_y_val in val_dl:
            #             batch_X_val = batch_X_val.to(device)
            #             batch_y_val = batch_y_val.to(device)

            #             outputs_val = []
            #             y_score = [] 
            #             for sample in range(BATCH_SIZE):
            #                 out = model(batch_X_val[sample])
            #                 outputs_val.append(out)
            #                 y_score.append(round(out.item()))

            #             val_loss = loss_fn(torch.stack(outputs_val).to(device), batch_y_val)
            #             avg_val_loss.append(val_loss.item())

            #         y_true = batch_y_val.data.cpu().numpy()
            #         acc = accuracy_score(batch_y_val.data.cpu().numpy(), y_score) * 100
            #         epoch_acc.append(acc)
            #         # print("Batch acc: " + acc)

            #     val_writer.plot_loss(val_loss, i, loss_name=f'loss_epoch_{epoch}')

        # Epoch validation

        y_score = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_dl:
                batch_X_val = batch_X_val.to(device)
                batch_y_val = batch_y_val.to(device)

                outputs_val = []
                for sample in range(BATCH_SIZE):
                    out = model(batch_X_val[sample])
                    outputs_val.append(out)
                    y_score.append(round(out.item()))

                val_loss = loss_fn(torch.stack(outputs_val).to(device), batch_y_val)
                avg_val_loss.append(val_loss.item())
                y_true += batch_y_val.data.cpu().numpy().tolist()

        acc = accuracy_score(y_true, y_score)
        acc_writer.plot_loss(acc, epoch, loss_name='avg_acc')
        print("Epoch acc: ", str(acc))

        print("Original acc: ", str(accuracy_score(y_true[0:1999], y_score[0:1999])))
        print("Fake acc: ", str(accuracy_score(y_true[1000:2999], y_score[1000:2999])))

        exp_lr_scheduler.step()
        
        train_writer.plot_loss(loss, epoch, loss_name='traingin_loss')
        val_writer.plot_loss(val_loss, epoch, loss_name='traingin_loss')
        train_writer.plot_loss(sum(avg_loss) / len(avg_loss), epoch, loss_name='avg_traingin_loss')
        val_writer.plot_loss(sum(avg_val_loss) / len(avg_val_loss), epoch, loss_name='avg_traingin_loss')
        # acc_writer.plot_loss(sum(epoch_acc) / len(epoch_acc), epoch, loss_name='avg_acc')
        # print("Epoch acc: " + str(sum(epoch_acc) / len(epoch_acc)))

        torch.save(model.module.state_dict(), f'{model_name}/model-epoch-{epoch}.ckpt')