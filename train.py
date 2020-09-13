import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import os
import tensorflow as tf
from sklearn.model_selection import KFold
import cv2

from utils import seed_everything, encode_feature
from models import build_model
from datasets import OSIC_Dataset


# Set GPU(s) to use
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="b1")
parser.add_argument("--num_folds", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)

parser.add_argument("--meta_train", type=str,
                    default="../data/small_train.csv")
parser.add_argument("--meta_test", type=str, default="../data/small_test.csv")
parser.add_argument("--data", type=str, default="../data/train")
parser.add_argument("--external_data", type=str, default="../data/img/")

parser.add_argument("--seed", type=int, default=1512)

args = parser.parse_args()
seed_everything(args.seed)

logging.basicConfig(filename=f"../logs/efficient_net_{args.model_name}.txt",
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info("Experiment config")
logging.info(args.__dict__)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

train = pd.read_csv(args.meta_train)
A = {}
features = {}
P = []

pbar = tqdm(enumerate(train.Patient.unique()), desc="Encoding features")
for i, p in pbar:
    # pbar.set_postfix(PATIENT=p)
    sub = train.loc[train.Patient == p, :]
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    a, b = np.linalg.lstsq(c, fvc)[0]

    A[p] = a
    features[p] = encode_feature(sub)
    P.append(p)

x, y = [], []
pbar = tqdm(train.Patient.unique(), desc="Getting mask")
for p in pbar:
    # pbar.set_postfix(PATIENT=p)
    try:
        ldir = os.listdir(args.external_data + f'mask_noise/mask_noise/{p}/')
        numb = [float(i[:-4]) for i in ldir]
        for i in ldir:
            x.append(cv2.imread(args.external_data +
                                f'mask_noise/mask_noise/{p}/{i}', 0).mean())
            y.append(float(i[:-4]) / max(numb))
    except Exception as e:
        logging.warning(e)
        logging.warning(f"Cannot find external data for patient {p}")
        pass

kf = KFold(n_splits=args.num_folds, random_state=args.seed, shuffle=False)
P = np.array(P)
subs = []

folds_history = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(P)):
    print('#####################')
    print('####### Fold %i ######' % fold)
    print('#####################')
    print('Training...')

    er = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=10,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    cpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='fold-%i.h5' % fold,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )

    rlp = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-8
    )
    model = build_model(model_class=args.model_name)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=args.lr), loss="mae")
    history = model.fit_generator(OSIC_Dataset(root=args.data,
                                               keys=P[tr_idx],
                                               a=A,
                                               features=features),
                                  steps_per_epoch=32,
                                  validation_data=OSIC_Dataset(root=args.data,
                                                             keys=P[val_idx],
                                                             a=A,
                                                             features=features),
                                  validation_steps=16,
                                  callbacks=[cpt, rlp],
                                  epochs=args.num_epochs)
    folds_history.append(history.history)
    print('Training done!')
