import tensorflow as tf
import numpy as np
from utils import get_img
import os
import logging


class OSIC_Dataset(tf.keras.utils.Sequence):
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, root, keys, a, features, batch_size=8):
        self.root = root
        # self.patient_ids = patient_ids
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.features = features
        self.batch_size = batch_size
        self.data = {}
        for p in self.keys:
            self.data[p] = os.listdir(self.root + f'/{p}/')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = []
        a, feature = [], []
        keys = np.random.choice(self.keys, size=self.batch_size)
        for k in keys:
            try:
                i = np.random.choice(self.data[k], size=1)[0]
                img = get_img(self.root+f'/{k}/{i}')
                x.append(img)
                a.append(self.a[k])
                feature.append(self.features[k])
            except Exception as e:
                logging.warning(e)
                logging.warning(self.root+f'/{k}/{i}')
                logging.warning(k, i)

        x, a, feature = np.array(x), np.array(a), np.array(feature)
        x = np.expand_dims(x, axis=-1)
        return [x, feature], a
