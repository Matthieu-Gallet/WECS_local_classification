import sys

sys.path.append("../")

from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)
from collections import Counter
from sklearn.utils import shuffle
import os, glob, time


def prepare_data(x, bands):
    x = x[:, :, :, bands]
    x = x.reshape(x.shape[0], -1)
    return x


if __name__ == "__main__":
    pathd = glob.glob("../../../results_II/aweig/*.h5")
    for inputh_data in pathd:
        print(inputh_data)

        output_dir = f"../results/{str(time.time())}/"
        bands = [0, 1, 2]
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(42)

        log = init_logger(output_dir)
        log.info("Start")
        log.info(f"Input data: {inputh_data}")
        log.info(f"Output dir: {output_dir}")
        log.info(f"Bands: {bands}")

        x, y = load_h5(inputh_data)
        # Try here to remove the non change class

        x = prepare_data(x, bands)
        # # x = Hist_SAR().fit_transform(x)
        log.info(f"x: {x.shape}")
        log.info(f"y: {y.shape}")
        log.info(f"y: {Counter(y)}")

        le = LabelEncoder()
        y = le.fit_transform(y)
        log.info(f"Classes: {le.classes_}")

        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        knn_cv = KNeighborsClassifier(
            n_neighbors=10, n_jobs=-1
        )  # , weights="distance", metric=KL_compound_gaussian)
        knn_cv = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, criterion="entropy"
        )
        f1, auc, kappa = [], [], []
        cm = []

        for i, (train_index, test_index) in enumerate(skf.split(x, y)):
            train_index = shuffle(train_index)
            test_index = shuffle(test_index)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print(x_test.shape, y_test.shape, x_train.shape, y_train.shape)
            log.info(f"Fold {i}:")
            log.info(f"x_train: {x_train.shape}")
            log.info(f"y_train: {y_train.shape}, {Counter(y_train)}")
            log.info(f"x_test: {x_test.shape}")
            log.info(f"y_test: {y_test.shape}, {Counter(y_test)}")

            knn_cv.fit(x_train, y_train)
            y_pred = knn_cv.predict(x_test)
            y_prob = knn_cv.predict_proba(x_test)

            log.info("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
            auc.append(
                roc_auc_score(y_test, y_prob, average="weighted", multi_class="ovr")
            )
            log.info("AUC: {:.2f}".format(auc[-1]))
            f1.append(f1_score(y_test, y_pred, average="weighted"))
            log.info("F1: {:.2f}".format(f1[-1]))
            kappa.append(cohen_kappa_score(y_test, y_pred))
            log.info("Kappa: {:.2f}".format(kappa[-1]))
            log.info("Confusion matrix:")
            cm.append(confusion_matrix(y_test, y_pred))
            log.info(f"\r{cm[-1]}")
            log.info("Classification report:")
            log.info(
                f"\r {classification_report(y_test, y_pred, target_names=le.classes_)}"
            )
            log.info("")
            print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Do wathever you want with the results (plot, save, etc...)
        log.info(f"Mean F1: {np.mean(f1)} +/- {np.std(f1)}")
        print(f"Mean F1: {np.mean(f1)} +/- {np.std(f1)}")
        log.info(f"Mean AUC: {np.mean(auc)} +/- {np.std(auc)}")
        log.info(f"Mean Kappa: {np.mean(kappa)} +/- {np.std(kappa)}")
        log.info(f"Mean Confusion matrix: \r {np.mean(cm, axis=0)}")
        log.info("End")
