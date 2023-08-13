import os, glob, sys
import numpy as np

sys.path.append("../")

from utils import *
from data.process.dataset_loader import prepare_data, balance_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

if __name__ == "__main__":
    type_an_G = ["-1", "_7", "63"]
    path_G = [
        f"../../../results_II/aweig/*-1*.h5",
        f"../../../results_II/win_anII/*_7*.h5",
        f"../../../results_II/win_anII/*63*.h5",
    ]

    times_name = str(time.time()).split(".")[0]
    output_dir = f"../results/multi_Glob_7_63_knn_II_{times_name}/"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(path_G)):
        type_an = type_an_G[i]
        path = path_G[i]
        inputh_data = glob.glob(path)[0]

        print(inputh_data)
        X, Y, le = prepare_data(
            inputh_data, band_max=[0, 1, 2], balanced=False, binary=False
        )
        print(X.shape, Y.shape)
        print(np.unique(Y, return_counts=True))
        print(le.classes_, le.transform(le.classes_))

        conf = []
        rng = np.random.default_rng(42)
        seed = rng.integers(0, 1000, 10)

        for s in seed:
            strat = StratifiedKFold(n_splits=4, shuffle=True, random_state=s)

            for train, test in strat.split(X, Y):
                x_train = X[train]
                y_train = Y[train]
                x_test = X[test]
                y_test = Y[test]

                x_train_B, y_train_B = balance_dataset(
                    x_train, y_train, shuffle=True, oversampling=True, seed=s
                )
                print(np.unique(y_train_B, return_counts=True))

                knn = KNeighborsClassifier(
                    n_neighbors=50, n_jobs=-1, weights="distance"
                )
                knn.fit(x_train_B, y_train_B)
                y_pred = knn.predict(x_test)
                f1_multi = 100 * confusion_matrix(
                    y_test,
                    y_pred,
                    labels=le.transform(le.classes_),
                    normalize="true",
                )

                print(
                    pd.DataFrame(
                        f1_multi.round(4),
                        index=le.classes_,
                        columns=le.classes_,
                    )
                )
                conf.append(f1_multi.round(4))
                print("-" * 50)
        conf = [np.array(conf), le.classes_]
        dump_pkl(conf, os.path.join(output_dir, f"multi_f1_{type_an}_{times_name}.pkl"))
