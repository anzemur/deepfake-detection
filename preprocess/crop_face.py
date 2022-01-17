import json
import os
import subprocess
import pathlib

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

from models.retina import Retina


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

def get_face(videoPath, save_root, method='mtcnn', select_nums=10, save_face=True):
    v_cap = cv2.VideoCapture(videoPath)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if select_nums == 'I':
        frame_types = get_frame_types(videoPath)
        samples = [x[0] for x in frame_types if x[1]=='I']
    else:
        if v_len > select_nums and select_nums != 0:
            samples = np.linspace(0, v_len - 1, select_nums).round().astype(int)
        else:
            samples = np.linspace(0, v_len - 1, v_len).round().astype(int)

    for numFrame in samples:
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, numFrame)
        _, vframe = v_cap.read()
        height, width = vframe.shape[:2]
        s = str(numFrame).zfill(3)

        try:
            if method == 'retina':
                boxes, probs, points = retina.detect_faces(vframe)
            else:
                image = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                boxes, probs, points = mtcnn.detect(image, landmarks=True)
            maxi = np.argmax(boxes[:,2]-boxes[:,0]) #get largest face
            boxes = boxes[maxi]
            points = points[maxi]
            
            if save_root and not os.path.exists(save_root):
                os.makedirs(save_root)
            
            if save_face:
                if save_root:
                    x, y, size = get_boundingbox(boxes.flatten(), width, height, 1.3)
                    cropped_face = vframe[y:y + size, x:x + size]
                    cv2.imwrite(os.path.join(save_root, "%s.png") % s, cropped_face)
            else:
                basename = os.path.splitext(os.path.basename(videoPath))[0]
                if save_root:
                    outname = os.path.join(save_root,video_type+'-'+basename+'_'+s+'.png')
                    cv2.imwrite(outname, vframe)
                #input for json file
                d[video_type+'-'+basename+'_'+s]={'box':boxes.tolist(),'landms':points.ravel().tolist()}


        except Exception as e:
            #face not detected
            print(f'ERROR: \nThere was an error while detecting face in: {videoPath}\n Message: {e}', )

            if save_root:
                if save_face:
                    cv2.imwrite(os.path.join(save_root, "%s_noface.png") % s, vframe)
                else:
                    basename = os.path.splitext(os.path.basename(videoPath))[0]
                    outname = os.path.join(save_root, video_type+'-'+basename+'_'+s+'_noface.png')
                    cv2.imwrite(outname, vframe)

    v_cap.release()
    return v_len


if __name__ == '__main__':
    # Configuration parameters.
    MODE = 'test'                                                                                  # Which images do we want to obtain  - train or test.
    SAVE_FACE=False                                                                                 # True: save faces; False: only frames with json metadata

    # Database configuration parameters.
    ROOT_PATH = str(pathlib.Path().resolve())                                                       # Root path of the current working directory.
    DATABASE = 'Celeb-DF-v2'                                                                        # Name of the DeepFake dataset.
    VIDEO_ROOT = f'/hdd2/vol1/deepfakeDatabases/original_videos/{DATABASE}'                         # The base dir of DeepFake dataset.
    OUTPUT_PATH = f'/hdd2/vol1/deepfakeDatabases/anzem-cropped_videos/frames/{DATABASE}/{MODE}'     # Where to save cropped training faces.
    TXT_PATH = f'{ROOT_PATH}/Celeb-DF-v2-{MODE}-list.txt'                                           # The given train-list.txt or test-list.txt file.
    META_PATH = f'{OUTPUT_PATH}/I-frames_meta.json'                                                 # Where to save frames meta data - coords of faces.

    print(VIDEO_ROOT)
    print(OUTPUT_PATH)
    print(TXT_PATH)
    print(META_PATH)

    with open(TXT_PATH, "r") as f:
        data = f.readlines()
    d={}
    all_frames = 0


    # MTCNN Face detector
    mtcnn = MTCNN(device="cuda:0").eval()
    # Retina Face detector
    retina = Retina(threshold=0.9, device="cuda:0").eval()
    for line in tqdm(data): #data[:100] only subset
        video_name = line[2:-1]
        video_type = line[0:1]
        video_path = os.path.join(VIDEO_ROOT, video_name)
        save_dir = None
        if OUTPUT_PATH:
            if SAVE_FACE:
                save_dir = os.path.join(OUTPUT_PATH, "faces", video_name.split('.')[0]) #save faces
            else:
                save_dir = os.path.join(OUTPUT_PATH, "I-frames") #save frames with json meta data
        frames = get_face(video_path, save_dir, method='retina', select_nums='I', save_face=SAVE_FACE) #select_nums: num of frames: 0=all, 'I'=I-frames
        all_frames += frames
    print('done')
    print('#all frames: '+str(all_frames))

    if d:
        with open(META_PATH, 'w') as f:
            json.dump(d, f, indent=4)
