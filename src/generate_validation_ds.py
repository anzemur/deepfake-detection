import os
import random
import shutil

from tqdm import tqdm

if __name__ == '__main__':
  # Configuration parameters.
  MODE = 'train'
  SCALE = 0.9
  PAIRS = 9
  SIZE = 0.2

  # Database configuration parameters.
  DATASET = 'Celeb-DF-v2'
  ROOT_DATA_PATH = f'/hdd2/vol1/deepfakeDatabases/anzem-cropped_videos/patches/{DATASET}/{SCALE}/{PAIRS}'
  TRAIN_DATA_PATH = f'{ROOT_DATA_PATH}/train'
  OUT_DATA_PATH = f'{ROOT_DATA_PATH}/validation'

  if not os.path.exists(OUT_DATA_PATH):
    os.makedirs(OUT_DATA_PATH)

  img_names = sorted(os.listdir(TRAIN_DATA_PATH))
  labels = list(map(lambda x: int(x.split('-')[0]), img_names))

  neg_count = labels.count(0)
  pos_count = labels.count(1)

  neg_examples = random.sample(img_names[0:neg_count], int(neg_count * SIZE))
  pos_examples = random.sample(img_names[neg_count + 1:pos_count], int(pos_count * SIZE))

  validation_set = neg_examples + pos_examples
  for img_name in tqdm(validation_set):
    target = os.path.join(TRAIN_DATA_PATH, img_name)
    shutil.move(target, OUT_DATA_PATH)
