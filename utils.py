import os
import random
import numpy as np
import tensorflow as tf

import pydicom
import cv2


def seed_everything(seed=1512):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def encode_feature(df):
    vector = [(df.Age.values[0]-30)/30]

    if df.Sex.values[0].lower() == 'male':
        vector.append(0)
    else:
        vector.append(1)

    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0, 0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1, 1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0, 1])
    else:
        vector.extend([1, 0])
    return np.array(vector)


def get_img(path):
    img = pydicom.dcmread(path)
    img = cv2.resize((img.pixel_array - img.RescaleIntercept) /
                     (img.RescaleSlope * 1000), (512, 512))
    return img
