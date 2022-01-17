import os
import sys
from turtle import pos
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Xception, ResNet18
from utils import PatchDataset

import seaborn as sns
import matplotlib.pyplot as plt

# Acc: 0.7462
# AUC Score: 0.7462
# gamma: 0.2
if __name__ == '__main__':
    # Configuration parameters.
    SCALE = 0.9
    PAIRS = 4
    DATASET = 'Celeb-DF-v2'
    BATCH_SIZE = 1
    PATCHES_DIR = f'/hdd2/vol1/deepfakeDatabases/anzem-cropped_videos/patches/{DATASET}/{SCALE}/{PAIRS}'
    MODELS_DIR = '/home/anzem/deepfake-patch-detection/trained_models/'
    MODEL_NAME = 'xception-1642251146.5699189/model-epoch-5.ckpt'
    # MODEL_NAME = 'resnet18-1642095173.869416/model-epoch-0.ckpt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = PatchDataset(f'{PATCHES_DIR}/test', PAIRS)
    # labels = X.get_labels()
    test_dl = DataLoader(
        X,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=4,
        shuffle=True
    )

    accs = []
    aucs = []

    gammas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for gamma in gammas:
        # model = ResNet18(gamma).to(device)
        model = Xception(gamma).to(device)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, MODEL_NAME)), strict=False)
        model.eval()

        predictions = []
        gt = []
        with torch.no_grad():
            for i, (batch_X, batch_y) in enumerate(tqdm(test_dl)):
                batch_X = batch_X.to(device)

                for sample in range(BATCH_SIZE):
                    out = model(batch_X[sample])
                    # predictions.append(round(torch.sigmoid(out).item()))
                    predictions.append(round(out.item()))
                    gt.append(int(batch_y.item()))

                # if ((i + 1) % 1000 == 0):
                #     acc = accuracy_score(gt, predictions)
                #     print('Acc: ' + str(acc))


        print("Gamma: " + str(gamma))

        acc = accuracy_score(gt, predictions)
        print('Acc: ' + str(acc))
        accs.append(acc)

        score = roc_auc_score(gt, predictions)
        print('AUC Score: ' + str(score))
        aucs.append(score)


    sns.set_style("darkgrid")
    plt.figure("auc_acc")
    plt.plot(gammas, accs, color="c", label="Classification accuracy")
    plt.plot(gammas, aucs, color="b", label="AUC")
    plt.legend(loc='upper right', fancybox=True)
    plt.gca().set_ybound(-0.1, 1.1)
    plt.ylabel("Evaluation accuracy")
    plt.xlabel("Regularization parameter value")
    plt.savefig("./res/auc_acc.pdf")
