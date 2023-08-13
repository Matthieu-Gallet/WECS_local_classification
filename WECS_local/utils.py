from os.path import dirname, abspath, join, basename, exists
from os import system, remove, makedirs, rename
import glob, sys, h5py, logging, time
from datetime import datetime
from shutil import copyfile, rmtree
from scipy.stats import spearmanr
from yaml import safe_load
from tqdm import tqdm
import pickle

from sklearn.pipeline import Pipeline
import numpy as np


def scale_image(img):
    img = img.astype(float)
    img = img - img.min()
    img = img / img.max() - img.min()
    return img


def load_yaml(file_name):
    with open(file_name, "r") as f:
        opt = safe_load(f)
    return opt


def exist_create_folder(path):
    if not exists(path):
        makedirs(path)
    return 1


def dump_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return 1


def open_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def init_logger(path_log):
    now = datetime.now()
    namlog = now.strftime("%d%m%y_%HH%MM%S")
    datestr = "%m/%d/%Y-%I:%M:%S %p "
    logging.basicConfig(
        filename=join(path_log, f"log_{namlog}.log"),
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        format="%(asctime)s: (%(filename)s): %(levelname)s: %(funcName)s Line: %(lineno)d - %(message)s",
    )
    logging.info("Started")
    return logging


def parser_pipeline(opt, ind):
    for imp in opt["import"]:
        exec(imp)
    pipe = opt["pipeline"][ind]
    step = []
    for i in range(len(pipe)):
        name_methode = pipe[i][0]
        estim = locals()[name_methode]()

        if len(pipe[i]) > 1:
            [
                [
                    setattr(estim, param, pipe[i][g][param])
                    for param in pipe[i][g].keys()
                ]
                for g in range(1, len(pipe[i]))
            ]
        step.append((name_methode, estim))
    return Pipeline(step, verbose=True)  # , memory=".cache")


def save_h5(img, label, filename):
    if ".h5" not in filename:
        filename += ".h5"
    with h5py.File(filename, "w") as hf:
        hf.create_dataset(
            "img", np.shape(img), h5py.h5t.IEEE_F32BE, compression="gzip", data=img
        )  # IEEE_F32BE is big endian float32
        hf.create_dataset(
            "label", np.shape(label), compression="gzip", data=label.astype("S")
        )


def load_h5(filename):
    if ".h5" not in filename:
        filename += ".h5"
    with h5py.File(filename, "r") as hf:
        data = np.array(hf["img"][:]).astype(np.float32)
        meta = np.array(hf["label"][:]).astype(str)
    return data, meta


def random_shuffle(X, y, rng=-1):
    """Shuffle randomly the dataset

    Parameters
    ----------
    X : numpy array
        dataset of images

    y : numpy array
        dataset of labels

    rng : int, optional
        Random seed, by default -1, must be a np.random.default_rng() object

    Returns
    -------
    numpy array
        shuffled dataset of images

    numpy array
        shuffled dataset of labels

    """
    if rng == -1:
        rng = np.random.default_rng(42)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y
