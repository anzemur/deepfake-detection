import random
import time
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils import crop_face, detect_face


def simulate_read_img():
  img = cv.imread("./data/person.jpeg")
  face_loc = detect_face(img)

  return img, face_loc

def generate_patch_pairs(img, face_loc, pairs_n=9):
  """
  Generates patch pairs in given image. 

  Args:
    img: Image to patch with specified face location.
    face_loc: Location of the face on the given image.
    patches_n: Number of patch pairs to generate.

  Returns:
    Tuple array of generated patch pairs: (face_patch, random_patch)
  """
  fi_h = face_loc['height']
  fi_w = face_loc['width']
  face_img = crop_face(img, face_loc)

  patches_per_row = int(np.sqrt(pairs_n))
  patch_w = fi_w//patches_per_row
  patch_h = fi_h//patches_per_row

  fi_w = patch_w * patches_per_row
  fi_h = patch_h * patches_per_row
  face_img = cv.resize(face_img, (fi_w, fi_h))

  face_patches = [
    face_img[x:x+patch_w, y:y+patch_h]
      for x in range(0, fi_w, patch_w)
      for y in range(0, fi_h, patch_h)
  ]

  pairs = []
  patches_coords = []
  for face_patch in face_patches:
    x, y = generate_random_patch(img, face_loc, patch_w, patch_h, patches_coords)
    patches_coords.append((x, y))
    pairs.append((face_patch, img[y:y + patch_h, x:x + patch_w]))

  return pairs
  
def generate_random_patch1(img, face_loc, patch_w, patch_h, patches_coords, no_overlap=True):
  """
  Generates random patch from the given image. Random patch should not overlap with the
  location of the face and with the locations of other patches (if `no_overlap` is set to True).

  Args:
    img: Image to generate patch from.
    face_loc: Location of the face on the given image.
    patch_w: Width of the patch.
    patch_h: Height of the patch.
    patches_coords: Coordinates of the existing patches.
    no_overlap: Tells if the patch should not overlap with the existing patches.

  Returns:
    Top left coordinates of the newly generated patch: x, y
  """
  img_h, img_w, _ = img.shape
  x = random.randint(0, img_w - patch_w - 1)
  y = random.randint(0, img_h - patch_h - 1)

  unique = True
  if patches_overlaps(x, x+patch_w, y, y+patch_h, face_loc['x'], face_loc['x']+face_loc['width'], face_loc['y'], face_loc['y']+face_loc['height']):
    unique = False

  if unique and no_overlap:
    for x1, y1 in patches_coords:
      if patches_overlaps(x, x+patch_w, y, y+patch_h, x1, x1+patch_w, y1, y1+patch_h):
        unique = False
        break

  if not unique:
    return generate_random_patch(img, face_loc, patch_w, patch_h, patches_coords, no_overlap)
  else:
    return x, y

def generate_random_patch(img, face_loc, patch_w, patch_h, patches_coords, no_overlap=False):
  """
  Generates random patch from the given image. Random patch should not overlap with the
  location of the face and with the locations of other patches (if `no_overlap` is set to True).

  Args:
    img: Image to generate patch from.
    face_loc: Location of the face on the given image.
    patch_w: Width of the patch.
    patch_h: Height of the patch.
    patches_coords: Coordinates of the existing patches.
    no_overlap: Tells if the patch should not overlap with the existing patches.

  Returns:
    Top left coordinates of the newly generated patch: x, y
  """
  img_h, img_w, _ = img.shape

  # cv.rectangle(img, (face_loc['x'], face_loc['y']), (face_loc['x']+face_loc['width'], face_loc['y']+face_loc['height']), (0,255,0), 2)

  # print("img_w:", img_w)
  # print("img_h:", img_h)
  # print("patch_w:", patch_w)
  # print("patch_h:", patch_h)

  unique = False
  count = 0
  while True:
    count += 1
    x = random.randint(0, img_w - patch_w - 1)
    y = random.randint(0, img_h - patch_h - 1)
    unique = True

    if patches_overlaps(x, x+patch_w, y, y+patch_h, face_loc['x'], face_loc['x']+face_loc['width'], face_loc['y'], face_loc['y']+face_loc['height']):
      # print("face overlaps")
      # print("x:", x)
      # print("y:", y)
      # print("patch_w:", patch_w)
      # print("patch_h:", patch_h)
      # print("overlap: ", count)
      unique = False

    if unique and no_overlap:
      for x1, y1 in patches_coords:
        if patches_overlaps(x, x+patch_w, y, y+patch_h, x1, x1+patch_w, y1, y1+patch_h):
          unique = False
          break

    if unique:
      cv.rectangle(img, (x, y), (x + patch_w, y + patch_h), (0,0,255), 2)
    

    # print("writing")
    # cv.imwrite(f"test-{time.time()}.png", img)
    # return x, y

    if unique:
      return x, y


def patches_overlaps(x1_l, x1_r, y1_t, y1_d, x2_l, x2__r, y2_t, y2_d):
  """
  Checks if two given patches overlap.
  Returns:
    True if patches overlaps, False othervise.
  """
  return (x1_r >= x2_l and x2__r >= x1_l) and (y1_d >= y2_t and y2_d >= y1_t)

def plot_pairs(pairs):
  fig = plt.figure()
  cols = len(pairs)

  for i in range(0, cols):
    fig.add_subplot(1, cols, i + 1)
    plt.imshow(pairs[i][0])

    fig.add_subplot(2, cols, i + 1)
    plt.imshow(pairs[i][1])

  plt.savefig(f"res/pairs-{time.time()}.png")
