
import numpy as np
import shutil
import copy
import os


def nor(frames):
    new_frames = frames.astype(np.float32)/255.0
    return new_frames

def de_nor(frames):
    new_frames = copy.deepcopy(frames)
    new_frames *= 255.0
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def normalization(frames,up=80):
    new_frames = frames.astype(np.float32)
    new_frames /= (up/2)
    new_frames -= 1
    return new_frames

def denormalization(frames,up=80):
    new_frames = copy.deepcopy(frames)
    new_frames += 1
    new_frames *= (up/2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def clean_fold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)