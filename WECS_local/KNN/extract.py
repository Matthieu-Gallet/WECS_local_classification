from utils import *
from geo_tools import *
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

classes_dict = {
    0: "No Change",
    1: "Sugar cane",
    2: "Pasture and fodder",
    3: "Market gardening",
    4: "Greenhouse crops or shadows",
    5: "Orchards",
    6: "Wooded areas",
    7: "Moor and Savannah",
    8: "Rocks and natural bare soil",
    9: "Relief shadows",
    10: "Water",
    11: "Urbanized areas",
}


def most_frequent_values(tab):
    """Returns the most frequent value of each sample in the array.

    Parameters
    ----------
    tab : np.array
        Array of shape (n, m, p) where n is the number of samples, m and p are the
        dimensions of the image.

    Returns
    -------
    np.array
        Array of shape (n,) containing the most frequent value of each sample.
    """
    y = []
    for i in tab:
        coun = Counter(i.flatten())
        y.append(coun.most_common(1)[0][0])
    return np.array(y)


def clean_dataset(dataset, y):
    """Removes samples with nan values.

    Parameters
    ----------
    dataset : np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands.

    y : np.array
        Array of shape (n,) containing the labels of each sample.

    Returns
    -------
    np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands. And the labels associated
    """
    has_nan = np.logical_or(np.isnan(dataset), dataset == -999).any(axis=(1, 2, 3))
    mask = np.logical_not(has_nan)
    x_clean = dataset[mask]
    y_clean = y[mask]
    return x_clean, y_clean


def remove_classes(dataset, y, classes):
    """Removes samples with labels in classes.

    Parameters
    ----------
    dataset : np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands.
    y : np.array
        Array of shape (n,) containing the labels of each sample.
    classes : list
        List of classes to remove.

    Returns
    -------
    np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands. And the labels associated

    """
    mask = np.logical_not(np.isin(y, classes))
    x_clean = dataset[mask]
    y_clean = y[mask]
    return x_clean, y_clean


def group_labels(labels):
    """Groups labels into 5 classes. The classes are:
        - 0: no change
        - 1: agricultural land
        - 2: forest
        - 3: bare ground plain
        - 4: water
        - 5: urbanized area

    Parameters
    ----------
    labels : np.array
        Array of shape (n,) containing the labels of each sample.

    Returns
    -------
    np.array
        Array of shape (n,) containing the labels of each sample.
    """
    labels = np.where(
        (labels == 1) | (labels == 2) | (labels == 4) | (labels == 3), 21, labels
    )
    labels = np.where((labels == 5) | (labels == 6), 22, labels)
    labels = np.where((labels == 7) | (labels == 8) | (labels == 9), 23, labels)
    labels = np.where(labels == 10, 24, labels)
    labels = np.where((labels == 11), 25, labels)

    new_labels = np.zeros(labels.shape, dtype="<U20")
    new_labels = np.where(labels == 0, "no change", new_labels)
    new_labels = np.where(labels == 21, "agricultural land", new_labels)
    new_labels = np.where(labels == 22, "forest", new_labels)
    new_labels = np.where(labels == 23, "bare ground plain", new_labels)
    new_labels = np.where(labels == 24, "water", new_labels)
    new_labels = np.where(labels == 25, "urbanized area", new_labels)
    return new_labels


def hist_labels(x_clean, y_clean, output_path, bins=20):
    """Plots the histogram of each band for each class.

    Parameters
    ----------
    x_clean : np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands.
    y_clean : np.array
        Array of shape (n,) containing the labels of each sample.
    output_path : str
        Path to save the histogram.
    bins : int, optional
        Number of bins, by default 20

    Returns
    -------
    None
    """
    f, ax = plt.subplots(2, 2, figsize=(15, 15))
    band = ["VV", "VH", "VV/VH", "L2(VV,VH)"]
    for j in range(4):
        for i in np.unique(y_clean):
            g = x_clean[y_clean == i][:, :, :, j].flatten()
            ax[j // 2, j % 2].hist(g, bins=bins, alpha=0.5, label=i, density=True)
        ax[j // 2, j % 2].legend()
        ax[j // 2, j % 2].set_title(f"band {band[j]}")
    plt.savefig(output_path + "_hist.pdf")


def subsample(X, Y, n_samples):
    """Subsamples the dataset to have the same number of samples for each class.

    Parameters
    ----------
    X : np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands.
    Y : np.array
        Array of shape (n,) containing the labels of each sample.
    n_samples : int
        Number of samples to keep for each class.

    Returns
    -------
    np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands. And the labels associated
    """

    counter = Counter(Y)
    indices = {label: np.where(Y == label)[0] for label in counter.keys()}

    indices_subsampled = {}
    for label, label_indices in indices.items():
        if len(label_indices) > n_samples:
            indices_subsampled[label] = np.random.choice(
                label_indices, n_samples, replace=False
            )
        else:
            indices_subsampled[label] = label_indices

    indices_subsampled = np.concatenate(list(indices_subsampled.values()))
    X_subsampled = X[indices_subsampled]
    Y_subsampled = Y[indices_subsampled]
    return X_subsampled, Y_subsampled


def balanced_sample(X, Y):
    """Subsamples the dataset to have the same number of samples for each class.

    Parameters
    ----------
    X : np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands.

    Y : np.array
        Array of shape (n,) containing the labels of each sample.

    Returns
    -------
    np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the image and q is the number of bands. And the labels associated
    """
    counter = Counter(Y)

    # We will now under-sample each class to have the same number of samples as the smallest class
    min_samples = min(counter.values())
    X_subsampled = []
    Y_subsampled = []

    # For each class, we under-sample it to have the same number of samples as the smallest class
    for label, _ in zip(np.unique(Y), counter.values()):
        X_label = X[Y == label]
        Y_label = Y[Y == label]
        X_label_subsampled, Y_label_subsampled = subsample(
            X_label, Y_label, n_samples=min_samples
        )
        X_subsampled.append(X_label_subsampled)
        Y_subsampled.append(Y_label_subsampled)

    # We concatenate the under-sampled data and labels for all classes
    Xn = np.concatenate(X_subsampled)
    Yn = np.concatenate(Y_subsampled)
    return Xn, Yn


def create_dataset(
    wind_gt,
    img,
    winsize,
    step,
    remove_c=[],
):
    """Creates the dataset from the images and the ground truth.

    Parameters
    ----------
    wind_gt : np.array
        Array of shape (n, m) containing the ground truth.
    img : np.array
        Array of shape (n, m, p) where n and m are the dimensions of the image and p is
        the number of bands.
    winsize : int
        Size of the window.
    step : int
        Step between two windows.
    remove_c : list, optional
        List of classes to remove, by default []

    Returns
    -------
    np.array
        Array of shape (n, m, p, q) where n is the number of samples, m and p are the
        dimensions of the patchs and q is the number of bands. And the labels associated
    """
    fenetres = []
    for k in range(img.shape[-1]):
        fenetres.append(
            np.lib.stride_tricks.sliding_window_view(img[:, :, k], (winsize, winsize))[
                :: winsize + step, :: winsize + step
            ]
        )
    fenetres = np.moveaxis(np.array(fenetres), 0, -1)
    fenetres = fenetres.reshape(
        -1, fenetres.shape[2], fenetres.shape[3], fenetres.shape[4]
    )
    wind_gt = wind_gt.reshape(-1, wind_gt.shape[2], wind_gt.shape[3])
    y = most_frequent_values(wind_gt)
    x_clean, y_clean = clean_dataset(fenetres, y)
    y_clean = group_labels(y_clean)
    x_clean, y_clean = remove_classes(x_clean, y_clean, remove_c)
    x_balanced, y_balanced = balanced_sample(x_clean, y_clean)
    return x_balanced, y_balanced


def create_name(i, output_path, winsize, step):
    idxn = basename(i).find("0801")
    name = basename(i)[:idxn]
    save_path = join(output_path, name + f"_w{winsize}_s{step}")
    return save_path


def save_infos(
    save_path,
    x_balanced,
    y_balanced,
    path,
    pathgt,
    winsize,
    step,
    classes_dict,
    remove_c,
):
    with open(save_path + "_infos.txt", "w") as f:
        f.write(f"windows size: {winsize}\n")
        f.write(f"step: {step}\n")
        f.write(f"number of classes: {len(np.unique(y_balanced))}\n")
        f.write(f"number of samples: {len(y_balanced)}\n")
        f.write(f"number of samples per class: {Counter(y_balanced)}\n")
        f.write(f"data shape: {x_balanced.shape}\n")
        f.write(f"min value: {np.min(x_balanced)}\n")
        f.write(f"max value: {np.max(x_balanced)}\n")
        f.write(f"input bands: {['VV', 'VH', 'VV/VH', 'L2(VV,VH)']} \n")
        f.write(f"input path: {path}\n")
        f.write(f"ground truth path: {pathgt}\n")
        f.write(f"output path: {save_path}\n")
        f.write(f"removed classes: {remove_c}\n")
        f.write(f"classes dict: {classes_dict}\n")


if __name__ == "__main__":
    output_path = "../dataset_E3/"
    winsize = 5
    step = 0
    bins = 100
    remove_c = ["water"]  # ,"urbanized area"]

    # Be careful between .tif and .tiff
    path = join("../dataset_E2/", "*.tif")
    pathgt = "../data/ground_truth/ground_truth.tif"

    gt = np.array(Image.open(pathgt))
    wind_gt = np.lib.stride_tricks.sliding_window_view(gt, (winsize, winsize))[
        :: winsize + step, :: winsize + step
    ]
    for i in tqdm(glob.glob(path)):
        img, _ = load_data(i)
        x_balanced, y_balanced = create_dataset(wind_gt, img, winsize, step, remove_c)
        save_path = create_name(i, output_path, winsize, step)
        save_infos(
            save_path,
            x_balanced,
            y_balanced,
            path,
            pathgt,
            winsize,
            step,
            classes_dict,
            remove_c,
        )
        save_h5(x_balanced, y_balanced, save_path)
        hist_labels(x_balanced, y_balanced, save_path, bins)
        print("#" * 50)
        print(f"saved at {save_path}")
        print("test load")
        x, y = load_h5(save_path)
        print(x.shape, y.shape)
        print(Counter(y))
        print("test load done")
        print("#" * 50)
