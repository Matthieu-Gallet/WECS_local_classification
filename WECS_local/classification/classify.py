from utils import *
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
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
import os


def scm(X):
    _, N = X.shape
    return (1 / N) * X @ X.T


import numpy.linalg as la


def prepare_data(x, bands):
    x = x[:, :, :, bands]
    x = x.reshape(x.shape[0], -1)
    return x


def Tyler(X):
    p, N = X.shape
    sigma = scm(X)
    for iter_ in range(20):
        sigma_inv = np.linalg.inv(sigma)
        tmp = np.einsum("ji, jl, li -> i", X, sigma_inv, X).reshape((-1, 1))
        tmp = 1 / tmp
        sigma = (1 / N) * X @ (tmp * X.T)
    return sigma


def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    eigvals, eigvects = np.linalg.eigh(Ci)
    eigvals = np.diag(operator(eigvals))
    Out = np.dot(np.dot(eigvects, eigvals), np.conjugate(eigvects).T)
    return Out


def invsqrtm(Ci):
    """Return the inverse matrix square root of a covariance matrix defined by :
    .. math::
            \mathbf{C} = \mathbf{V} \left( \mathbf{\Lambda} \\right)^{-1/2}
            \mathbf{V}^T
    where :math:`\mathbf{\Lambda}` is the diagonal matrix of eigenvalues
    and :math:`\mathbf{V}` the eigenvectors of :math:`\mathbf{Ci}`
    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root
    """

    def isqrt(x):
        return 1.0 / np.sqrt(x)

    return _matrix_operator(Ci, isqrt)


def tyler_estimator_covariance(x, tol=0.001, iter_max=20):
    """A function that computes the Tyler Fixed Point Estimator for covariance matrix estimation
    Inputs:
        * 𝐗 = a matrix of size p*N with each observation along column dimension
        * tol = tolerance for convergence of estimator
        * iter_max = number of maximum iterations
    Outputs:
        * 𝚺 = the estimate
        * δ = the final distance between two iterations
        * iteration = number of iterations til convergence"""

    # Initialisation
    (p, N) = x.shape
    δ = np.inf  # Distance between two iterations
    𝚺 = np.eye(p)  # Initialise estimate to identity
    iteration = 0
    mu = np.mean(x, axis=1, keepdims=True)  # Mean of the data
    # Recursive algorithm
    while (δ > tol) and (iteration < iter_max):
        # if np.linalg.det(𝚺) == 0:
        #     import pdb; pdb.set_trace()

        𝐗 = x - mu
        # Computing expression of Tyler estimator (with matrix multiplication)

        temp = invsqrtm(𝚺) @ X
        τ = np.einsum("ij,ji->i", temp.T, temp)
        # τ = np.diagonal(𝐗.conj().T @ np.linalg.inv(𝚺) @ 𝐗)

        𝐗_bis = 𝐗 / np.sqrt(τ)
        𝚺_new = (p / N) * 𝐗_bis @ 𝐗_bis.conj().T

        # Imposing trace constraint: Tr(𝚺) = p
        𝚺_new = p * 𝚺_new / np.trace(𝚺_new)

        # Condition for stopping
        δ = np.linalg.norm(𝚺_new - 𝚺, "fro") / np.linalg.norm(𝚺, "fro")
        iteration = iteration + 1

        # Updating 𝚺
        𝚺 = 𝚺_new
        mu = (1 / np.sum(1 / np.sqrt(τ))) * np.sum(x / np.sqrt(τ), axis=1).reshape(
            (-1, 1)
        )

    # if iteration == iter_max:
    #     warnings.warn('Recursive algorithm did not converge')
    return (𝚺, τ, mu, δ, iteration)


def KL_compound_gaussian(f, g):
    f = f.reshape(25, 2).T
    g = g.reshape(25, 2).T
    cov_est1, tau1, mean_est1, _, _ = tyler_estimator_covariance(
        f, tol=0.01, iter_max=20
    )
    cov_est2, tau2, mean_est2, _, _ = tyler_estimator_covariance(
        g, tol=0.01, iter_max=20
    )
    cov_est2 = (cov_est2 / np.trace(cov_est2)) * np.trace(scm(g))
    cov_est1 = (cov_est1 / np.trace(cov_est1)) * np.trace(scm(f))

    kl_dir = KLCG(cov_est1, cov_est2, mean_est1, mean_est2, tau1, tau2)
    kl_inv = KLCG(cov_est2, cov_est1, mean_est2, mean_est1, tau2, tau1)
    return 0.5 * (kl_dir + kl_inv)


def KLCG(cov_est1, cov_est2, mean_est1, mean_est2, tau1, tau2):
    n, p = tau1.shape[0], mean_est1.shape[0]
    inc_cov2 = la.inv(cov_est2)
    diff_mu = mean_est2 - mean_est1
    tr = np.trace(inc_cov2 @ cov_est1)
    a1 = np.sum(tau1 * tr / tau2)
    mahan = diff_mu.T @ inc_cov2 @ diff_mu
    a2 = np.sum(mahan / tau2)
    a3 = n * np.log(la.det(cov_est2) / la.det(cov_est1)) - n * p
    kl = 0.5 * (a1 + a2 + a3)
    return kl


if __name__ == "__main__":
    # from class_stats_SAR import Hist_SAR

    # inputh_data = "../dataset_E3/patchs_size_5/WECS_R0.875_11_4C__w5_s0"
    # inputh_data = "../dataset_E3/patchs_size_5/WECS_R0.5_11_4C__w5_s0"
    inputh_data = "../dataset_E3/patchs_size_5/WECS_R0.5_31_4C__w5_s0"
    # inputh_data = "../dataset_E3/patchs_size_5/WECS_R0.5_5_4C__w5_s0"
    # inputh_data = "../dataset_E3/patchs_size_5_GLOBAL/WECS_R_G_4C__w5_s0"
    output_dir = f"../results/{str(time.time())}/"
    bands = [0, 1, 2]  # , 2, 3]
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
    # x = Hist_SAR().fit_transform(x)
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
        auc.append(roc_auc_score(y_test, y_prob, average="weighted", multi_class="ovr"))
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
