import json
import os

import PIL.Image as Image

import cv2 as cv
from tqdm import tqdm

from utils import generate_patch_pairs, get_boundingbox, plot_pairs

if __name__ == '__main__':
  # Configuration parameters.
  MODE = 'test'
  SCALE = 0.9
  PAIRS = 4

  # Database configuration parameters.
  DATASET = 'Celeb-DF-v2'
  ROOT_DATA_PATH = '/hdd2/vol1/deepfakeDatabases/anzem-cropped_videos'
  FRAMES_PATH = f'{ROOT_DATA_PATH}/frames/{DATASET}/{MODE}/I-frames'
  META_PATH = f'{ROOT_DATA_PATH}/frames/{DATASET}/{MODE}/I-frames_meta.json'
  OUT_DATA_PATH = f'{ROOT_DATA_PATH}/patches/{DATASET}/{SCALE}/{PAIRS}/{MODE}'

  with open(META_PATH, "r") as f:
    faces_meta = json.load(f)

  if not os.path.exists(OUT_DATA_PATH):
    os.makedirs(OUT_DATA_PATH)

  data = []
  for frame_path in tqdm(os.listdir(FRAMES_PATH)):
    frame_name = frame_path.split('.')[0]
    frame_class = frame_name.split('-')[0]

    if frame_name not in faces_meta:
      continue
      
    frame_face = faces_meta[frame_name]['box']
    frame_img = cv.imread(os.path.join(FRAMES_PATH, frame_path))
    img_h, img_w, _ = frame_img.shape
    face_loc = get_boundingbox(frame_face, img_w, img_h, SCALE) # Try scaling after 
    pairs = generate_patch_pairs(frame_img, face_loc, PAIRS)

    outdir = os.path.join(f'{OUT_DATA_PATH}', frame_name)
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    for idx, (face_patch, patch) in enumerate(pairs):
      cv.imwrite(f'{outdir}/{str(idx)}_face.png', face_patch)
      cv.imwrite(f'{outdir}/{str(idx)}_patch.png', patch)
