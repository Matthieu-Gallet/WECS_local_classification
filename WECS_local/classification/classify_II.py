import os, glob, sys
import numpy as np

sys.path.append("../")

from utils import *
from data.process.dataset_loader import prepare_data
from sklearn.model_selection import StratifiedKFold
from data.process.model_selection import BFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


if __name__ == "__main__":
    type_an = "aweig"  # "win_anII"  #
    path = f"../../../results_II/{type_an}/*.h5"
    times_name = str(time.time()).split(".")[0]
    output_dir = f"../results/BFold_{type_an}_knn_II_{times_name}/"
    os.makedirs(output_dir, exist_ok=True)
    pathd = glob.glob(path)

    pathd.sort()
    knn_f1 = {}

    for inputh_data in pathd:
        weights = float(inputh_data.split("_")[2][1:])
        # weights = float(inputh_data.split("_")[4])

        print(inputh_data)
        X, Y, le = prepare_data(inputh_data, band_max=[0, 1, 2], balanced=False)
        print(X.shape, Y.shape)
        print(np.unique(Y, return_counts=True))
        print(le.classes_, le.transform(le.classes_))
        test = np.where(Y == np.argmin(np.unique(Y, return_counts=True)[1]))[0]

        strat = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        rf_kf, knn_kf = [], []

        for train, test in strat.split(X, Y):
            x_train = X[train]
            y_train = Y[train]
            x_test = X[test]
            y_test = Y[test]
            bfold = BFold(shuffle=True, random_state=42)
            for btr in bfold.split(x_train, y_train):
                x_btr = x_train[btr]
                y_btr = y_train[btr]
                print(np.unique(y_btr, return_counts=True))

                knn = KNeighborsClassifier(
                    n_neighbors=50, n_jobs=-1, weights="distance"
                )
                knn.fit(x_btr, y_btr)
                y_pred = knn.predict(x_test)
                f1_sc_k = f1_score(y_test, y_pred, average="weighted")
                print(f"f1 score knn: {f1_sc_k}")
                knn_kf.append(f1_sc_k)
                print("-" * 50)

        knn_f1[weights] = knn_kf

    dump_pkl(knn_f1, os.path.join(output_dir, f"BFOLD_knn_f1_{times_name}.pkl"))
