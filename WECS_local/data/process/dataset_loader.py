import numpy as np
import os, sys

sys.path.append("../")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import load_h5, random_shuffle


def balance_dataset(X, Y, shuffle=False, oversampling=False, seed=42):
    """Balance the dataset by taking the minimum number of samples per class (under-sampling)

    Parameters
    ----------
    X : numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands)

    Y : numpy array
        dataset of labels in string, shape (n_samples,)

    shuffle : bool, optional
        Shuffle the dataset, by default False

    Returns
    -------
    numpy array
        balanced dataset of images in float32, shape (n_samples, height, width, n_bands)

    numpy array
        balanced dataset of labels in string, shape (n_samples,)
    """
    if oversampling:
        X, Y = oversampling_minority(X, Y)
    if shuffle:
        X, Y = random_shuffle(X, Y, seed=seed)
    cat, counts = np.unique(Y, return_counts=True)
    min_count = np.min(counts)
    X_bal = []
    Y_bal = []
    for category in cat:
        idx = np.where(Y == category)[0]
        idx = idx[:min_count]
        X_bal.append(X[idx])
        Y_bal.append(Y[idx])
    X_bal = np.concatenate(X_bal)
    Y_bal = np.concatenate(Y_bal)
    return X_bal, Y_bal


def oversample_indices(indices, desired_size):
    """
    Oversamples the given indices to achieve the desired size.

    Parameters:
        indices (list or numpy array): Indices of the minority class samples.
        desired_size (int): Desired size after oversampling.

    Returns:
        oversampled_indices (numpy array): Array of oversampled indices.
    """
    num_indices = len(indices)
    oversample_factor = desired_size // num_indices

    # Duplicate the indices based on the oversample factor
    oversampled_indices = np.repeat(indices, oversample_factor)

    # Calculate the remaining number of samples needed to reach the desired size
    remaining_samples = desired_size - len(oversampled_indices)

    # Randomly sample the remaining indices to fill up to the desired size
    if remaining_samples > 0:
        remaining_indices = np.random.choice(indices, remaining_samples, replace=True)
        oversampled_indices = np.concatenate((oversampled_indices, remaining_indices))
    oversampled_indices = np.concatenate((oversampled_indices, indices))
    oversampled_indices = np.random.permutation(oversampled_indices)
    return oversampled_indices


def oversampling_minority(X, Y):
    unique, counts = np.unique(Y, return_counts=True)
    sort_counts = np.argsort(counts)
    diff_first_to_sec = counts[sort_counts[1]] - counts[sort_counts[0]]
    min_class = unique[sort_counts[0]]
    idx = np.where(Y == min_class)[0]
    idx_over = oversample_indices(idx, diff_first_to_sec)

    idx_maj = np.where(Y != min_class)[0]
    idx_bal = np.concatenate((idx_maj, idx_over))
    X_bal = X[idx_bal]
    Y_bal = Y[idx_bal]
    return X_bal, Y_bal


def prepare_data(
    path,
    band_max=[0, 1, 2],
    balanced=True,
    shuffle=True,
):
    """Prepare the data for the CNN model, it suppose that the data are stored in hdf5 files in the same folder and named "data_train.h5" and "data_test.h5"

    Parameters
    ----------
    ipath : str
        Path to the hdf5 files

    frac_val : float, optional
        Fraction of the dataset to use for validation, by default 0.15

    band_max : list, optional
        List of the bands to use, by default [0, 1, 6, 7]

    balanced : list, optional
        List of boolean to balance the dataset train and test, by default [False, False]

    shuffle : bool, optional
        Shuffle the dataset (seed=42), by default True

    Returns
    -------
    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the train set

    numpy array
        dataset of labels in string, shape (n_samples,) of the train set

    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the test set

    numpy array
        dataset of labels in string, shape (n_samples,) of the test set

    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the validation set

    numpy array
        dataset of labels in string, shape (n_samples,) of the validation set

    sklearn.preprocessing.LabelEncoder
        LabelEncoder object to transform the labels into integers
    """
    X, Y = load_h5(path)
    if band_max is not None:
        X = X[:, :, :, band_max]

    if shuffle:
        X, Y = random_shuffle(X, Y)
    Y = np.where(Y != "no change", "change", Y)

    if balanced:
        X, Y = balance_dataset(X, Y)

    Le = LabelEncoder()
    y = Le.fit_transform(Y)

    X = X.reshape(X.shape[0], -1)
    return X, y, Le


def prepare_data_split(
    path,
    band_max=[0, 1, 2],
    balanced=[False, False],
    shuffle=True,
):
    """Prepare the data for the CNN model, it suppose that the data are stored in hdf5 files in the same folder and named "data_train.h5" and "data_test.h5"

    Parameters
    ----------
    ipath : str
        Path to the hdf5 files

    frac_val : float, optional
        Fraction of the dataset to use for validation, by default 0.15

    band_max : list, optional
        List of the bands to use, by default [0, 1, 6, 7]

    balanced : list, optional
        List of boolean to balance the dataset train and test, by default [False, False]

    shuffle : bool, optional
        Shuffle the dataset (seed=42), by default True


    Returns
    -------
    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the train set

    numpy array
        dataset of labels in string, shape (n_samples,) of the train set

    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the test set

    numpy array
        dataset of labels in string, shape (n_samples,) of the test set

    numpy array
        dataset of images in float32, shape (n_samples, height, width, n_bands) of the validation set

    numpy array
        dataset of labels in string, shape (n_samples,) of the validation set

    sklearn.preprocessing.LabelEncoder
        LabelEncoder object to transform the labels into integers
    """
    X, Y = load_h5(path)

    if band_max is not None:
        X_train = X_train[:, :, :, band_max]
        X_test = X_test[:, :, :, band_max]

    if shuffle:
        X_train, Y_train = random_shuffle(X_train, Y_train)
        X_test, Y_test = random_shuffle(X_test, Y_test)

    Le = LabelEncoder()
    y_train = Le.fit_transform(Y_train)
    y_test = Le.transform(Y_test)

    if balanced[0]:
        X_train, y_train = balance_dataset(X_train, y_train)
    if balanced[1]:
        X_test, y_test = balance_dataset(X_test, y_test)

    return X_train, X_test, y_train, y_test, Le
