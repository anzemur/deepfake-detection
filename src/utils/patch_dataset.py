import os

import cv2 as cv
import PIL.Image as Image
import torch
import torchvision.transforms as Transforms
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, img_folder, n_patches, transforms=None, neg_split=None, pos_split=None):
      self.img_folder = img_folder
      self.n_patches = n_patches
      self.img_names = sorted(os.listdir(img_folder))

      # Negative and positive classes split for smaller balanced dataset creation.
      if neg_split and pos_split:
        neg_classes = self.img_names[neg_split[0]:neg_split[1]]
        pos_classes = self.img_names[pos_split[0]:pos_split[1]]
        self.img_names = neg_classes + pos_classes

      if transforms is None:
        self.transforms = Transforms.Compose([
          Transforms.Resize((80, 80)),
          Transforms.ToTensor(),
          Transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        # self.transforms = Transforms.Compose([
        #     Transforms.Resize((56, 56)),
        #     Transforms.ToTensor(),
        #     Transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

    def get_image_name(self, idx):
      return self.img_names[idx]

    def get_labels(self):
      return list(map(lambda x: int(x.split('-')[0]), self.img_names))

    def get_image_class(self, idx):
      return self.img_names[idx].split('-')[0]

    def __len__(self):
      return len(self.img_names)

    def get_balanced_classes_weights(self):
      targets = torch.tensor(self.get_labels())

      class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
      )
    
      weight = 1. / class_sample_count.float()
      samples_weight = torch.tensor([weight[t] for t in targets])
      return samples_weight

    def __getitem__(self, idx):
      img_name = self.img_names[idx]
      patch_dir = os.path.join(self.img_folder, img_name)
      patches = []

      for i in range(self.n_patches):
        face_patch = cv.imread(os.path.join(patch_dir, f'{str(i)}_face.png'))
        face_patch = cv.cvtColor(face_patch, cv.COLOR_BGR2RGB)
        face_patch = Image.fromarray(face_patch)
        face_patch = self.transforms(face_patch)
        face_patch = torch.unsqueeze(face_patch, dim=0)

        patch = cv.imread(os.path.join(patch_dir, f'{str(i)}_patch.png'))
        patch = cv.cvtColor(patch, cv.COLOR_BGR2RGB)
        patch = Image.fromarray(patch)
        patch = self.transforms(patch)
        patch = torch.unsqueeze(patch, dim=0)

        patches.append(torch.stack((face_patch, patch)))
        
        
        # [(face_patch, random_patch), (face_patch, random_patch), (face_patch, random_patch), (face_patch, random_patch), (face_patch, random_patch)]

      return torch.stack(patches), float(img_name.split('-')[0]) # try chaing label to tensor

if __name__ == '__main__':
  SCALE = 0.9
  PAIRS = 4
  DATASET = 'Celeb-DF-v2'
  PATCHES_DIR = f'/hdd2/vol1/deepfakeDatabases/anzem-cropped_videos/patches/{DATASET}/{SCALE}/{PAIRS}/validation'


  X = PatchDataset(PATCHES_DIR, PAIRS,  neg_split=[1000, 2000], pos_split=[7000, 8000])
  # X = PatchDataset(PATCHES_DIR, PAIRS)
  labels = X.get_labels()
  print(labels.count(1))
  print(labels.count(0))
