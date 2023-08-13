import os, glob, sys
import numpy as np

sys.path.append("../")

from utils import *
from data.process.dataset_loader import prepare_data, balance_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score


if __name__ == "__main__":
    type_an = "-1"  # "0.01"  #
    path = f"../../../results_II/aweig/*{type_an}*.h5"
    times_name = str(time.time()).split(".")[0]
    output_dir = f"../results/multi_{type_an}_knn_II_{times_name}/"
    os.makedirs(output_dir, exist_ok=True)
    inputh_data = glob.glob(path)

    weights = float(inputh_data.split("_")[2][1:])
    # weights = float(inputh_data.split("_")[4])

    print(inputh_data)
    X, Y, le = prepare_data(inputh_data, band_max=[0, 1, 2], balanced=False)
    print(X.shape, Y.shape)
    print(np.unique(Y, return_counts=True))
    print(le.classes_, le.transform(le.classes_))

    f1scmul = {}
    for i in le.classes_:
        f1scmul[i] = []
    f1scmul["weighted"] = []

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

            knn = KNeighborsClassifier(n_neighbors=50, n_jobs=-1, weights="distance")
            knn.fit(x_train_B, y_train_B)
            y_pred = knn.predict(x_test)
            f1_multi = f1_score(
                y_test, y_pred, average=None, labels=le.transform(le.classes_)
            )

            for i in le.transform(le.classes_):
                f1scmul[le.inverse_transform([i])[0]].append(f1_multi[i])

            f1_sc_k = f1_score(y_test, y_pred, average="weighted")
            f1scmul["weighted"].append(f1_sc_k)

            print(f"f1 score knn: {f1_sc_k}")
            print("-" * 50)
    dump_pkl(f1scmul, os.path.join(output_dir, f"multi_f1_{type_an}_{times_name}.pkl"))
