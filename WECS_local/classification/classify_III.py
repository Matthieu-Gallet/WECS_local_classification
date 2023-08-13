import os, glob, sys
import numpy as np

sys.path.append("../")

from utils import *
from data.process.dataset_loader import prepare_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    # path = "../../../results_II/aweig/*.h5"
    type_an = "win_anII"  # "aweig"
    path = f"../../../results_II/{type_an}/*.h5"
    times_name = str(time.time()).split(".")[0]
    output_dir = f"../results/{type_an}_knn_II_{times_name}/"
    os.makedirs(output_dir, exist_ok=True)
    pathd = glob.glob(path)

    pathd.sort()

    rf_f1, knn_f1 = {}, {}

    # inputh_data = pathd[0]
    for inputh_data in pathd:
        # weights = float(inputh_data.split("_")[2][1:])
        weights = float(inputh_data.split("_")[3])

        print(inputh_data)
        X, Y, le = prepare_data(inputh_data, band_max=[0, 1, 2], balanced=True)
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

            knn = KNeighborsClassifier(n_neighbors=50, n_jobs=-1)
            # rf = RandomForestClassifier(
            #     n_estimators=100, n_jobs=-1, criterion="entropy"
            # )

            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            f1_sc_k = f1_score(y_test, y_pred, average="macro")
            print(f"f1 score knn: {f1_sc_k}")
            knn_kf.append(f1_sc_k)

            # rf.fit(x_train, y_train)
            # y_pred = rf.predict(x_test)
            # f1_sc_r = f1_score(y_test, y_pred, average="macro")
            # print(f"f1 score rf: {f1_sc_r}")
            # rf_kf.append(f1_sc_r)

            print("-" * 50)

        knn_f1[weights] = knn_kf

        dump_pkl(knn_f1, os.path.join(output_dir, f"knn_f1_{times_name}.pkl"))
