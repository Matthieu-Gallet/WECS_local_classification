from geo_tools import *
from utils import *
import pywt, numba, time


def scale(im):
    images = np.where(im > -999, im, np.nan)
    scaled = (images - np.nanmin(images)) / (np.nanmax(images) - np.nanmin(images))
    # imsc = np.where(im>-999,scaled,-999)
    return scaled


def find_crop(h, w):
    sh = 1
    sw = 1

    while h % 4 != 0:
        h = h + sh
        sh = sh + 1

    while w % 4 != 0:
        w = w + sw
        sw = sw + 1

    return (h, w)


def extend_images(images):
    h, w = np.shape(images[0])
    n = len(images)

    power = np.ceil(np.log2(np.min([h, w])))
    two_powered = (2**power).astype(int)

    padding = np.ceil((two_powered - np.min([h, w])) / 2).astype(int)
    crop_h, crop_w = find_crop(padding + h, padding + w)

    extended = []
    for i in range(0, n):
        new_im = pywt.pad(images[i], padding, "periodization")
        cropped = new_im[0:crop_h, 0:crop_w]

        extended.append(cropped)

    return (np.array(extended), padding)


def calculate_wavelet_images(images):
    # image dimension has to be mutiple of 2^J (in this case J = 2)
    extended, padding = extend_images(images)
    h, w = np.shape(images[0])
    n = len(extended)

    x = []
    for i in range(0, n):
        wavelet = pywt.swtn(extended[i], wavelet="db2", level=2, trim_approx=True)
        approx_coeffs = np.array(wavelet[0])[
            padding : padding + h, padding : padding + w
        ]

        x.append(approx_coeffs)

    return np.array(x)


@numba.jit(nopython=True, parallel=True, cache=True)
def calculate_dmatrices(x_images, mean_image):
    n = len(x_images)
    d_matrices = np.empty((n, *x_images[0].shape))

    for i in range(n):
        d_matrices[i] = np.square(x_images[i] - mean_image)
    return d_matrices


####### OLD GLOBAL CORRELATION
# @numba.jit(nopython=True, parallel=True, cache=True)
# def calculate_R(d_matrices, d_vector):
#     h, w, d = d_matrices.shape
#     corr = np.zeros((h, w))

#     d_vector_mean = 0
#     d_vector_std = 0

#     for k in range(d):
#         d_vector_mean += d_vector[k]
#         d_vector_std += d_vector[k] * d_vector[k]

#     d_vector_mean /= d
#     d_vector_std = d_vector_std / d - d_vector_mean * d_vector_mean
#     d_vector_std = d_vector_std**0.5

#     for i in numba.prange(h):
#         for j in numba.prange(w):
#             mean = 0
#             std = 0

#             for k in range(d):
#                 mean += d_matrices[i, j, k]
#                 std += d_matrices[i, j, k] * d_matrices[i, j, k]

#             mean /= d
#             std = std / d - mean * mean
#             std = std**0.5

#             rij = 0
#             for k in range(d):
#                 rij += (d_matrices[i, j, k] - mean) * (d_vector[k] - d_vector_mean)

#             rij /= d
#             rij /= std * d_vector_std

#             corr[i, j] = rij #abs()

#     return corr

####### NEW ALPHA CORRELATION
@numba.jit(nopython=True, parallel=True, cache=True)
def calculate_R(d_matrices, d_vector, alpha, win):
    h, w, d = d_matrices.shape
    corr = np.zeros((h, w))

    d_vector_mean = 0
    d_vector_std = 0

    for k in range(d):
        d_vector_mean += d_vector[k]
        d_vector_std += d_vector[k] * d_vector[k]

    d_vector_mean /= d
    d_vector_std = d_vector_std / d - d_vector_mean * d_vector_mean
    d_vector_std = d_vector_std**0.5

    for i in numba.prange(h):
        for j in numba.prange(w):
            d_matrices_dash = d_matrices[
                max([i - win, 0]) : min([i + win + 1, h - 1]),
                max([j - win, 0]) : min([j + win + 1, w - 1]),
                :,
            ]

            d_dash = np.zeros(d)
            for k in range(d):
                d_dash[k] = np.sum(d_matrices_dash[:, :, k])
                d_dash[k] -= d_matrices_dash[i, j, k]

            d_dash_mean = 0
            d_dash_std = 0

            for k in range(d):
                d_dash_mean = d_dash_mean + d_dash[k]
                d_dash_std = d_dash_std + (d_dash[k] * d_dash[k])

            d_dash_mean /= d
            d_dash_std = d_dash_std / d - d_dash_mean * d_dash_mean
            d_dash_std = d_dash_std**0.5

            # local
            mean = 0
            std = 0
            for k in range(d):
                mean += d_matrices[i, j, k]
                std += d_matrices[i, j, k] * d_matrices[i, j, k]
            mean /= d
            std = std / d - mean * mean
            std = std**0.5

            rij = 0
            rij_dash = 0
            for k in range(d):
                rij += (d_matrices[i, j, k] - mean) * (d_vector[k] - d_vector_mean)
                rij_dash += (d_matrices[i, j, k] - mean) * (d_dash[k] - d_dash_mean)

            rij /= d
            rij /= std * d_vector_std

            rij_dash /= d
            rij_dash /= std * d_dash_std

            # corr[i, j] = alpha * abs(rij) + (1 - alpha) * abs(rij_dash)
            corr[i, j] = alpha * rij + (1 - alpha) * rij_dash
    return corr


# def calculate_R(d_matrices, d_vector):
#     h,w = np.shape(d_matrices[0])
#     R = np.zeros((h,w))

#     for i in range(0, h):
#         for j in range(0, w):
#             dij = d_matrices[:,i,j]
#             rij = np.corrcoef(dij, d_vector)[0,1]
#             R[i,j] = abs(rij)

#     return R


def read_time_series(path):
    images = []
    in_path = glob.glob(path)
    in_path.sort()
    # Reading images
    for file in tqdm(in_path):
        imarray, geo = load_data(file)
        images.append(imarray[:, :, :])
    images = np.array(images)
    return images, geo


def average_sc_img(img):
    images = np.where(img > -999, img, np.nan)
    avg = np.nanmean(images, axis=0)
    scaled = (avg - np.nanmin(avg)) / (np.nanmax(avg) - np.nanmin(avg))
    # imsc = np.where(np.isnan(avg),-999,avg)
    return scaled


def run_wecs(path, outpath, alphas, wins):

    timeseries, geo = read_time_series(path)
    name = ["VV", "VH", "VVVH", "S__VV2_VH2"]
    print(timeseries.shape)
    # show_images(images)
    Full = np.zeros((timeseries.shape[1], timeseries.shape[2], 4))
    for alpha in alphas:
        for win in tqdm(wins):
            wecs_pair(outpath, alpha, win, timeseries, geo, Full, name)

    return 1


def wecs_pair(outpath, alpha, win, timeseries, geo, Full, name):
    for i in range(timeseries.shape[-1]):
        # Reference image
        images = timeseries[:, :, :, i]

        mean_image = average_sc_img(images)

        # mean_image = scale(mean_image)
        images = scale(images)

        # print(np.nanmin(mean_image), np.nanmax(mean_image))
        # print(np.nanmin(images, axis=(1, 2)), np.nanmax(images, axis=(1, 2)))
        # array2raster(mean_image[:,:,np.newaxis], geo,f"../dataset_E2/mean_image{i}.tiff", gdal_driver="GTiff")

        # Wavelet images
        x_images = calculate_wavelet_images(images)

        # Squared deviations matrices
        d_matrices = calculate_dmatrices(x_images, mean_image)

        # Overall change
        d_vector = np.nansum(d_matrices, axis=(1, 2))

        # Correlation matrix
        R = calculate_R(d_matrices.T, d_vector, alpha, win).T
        print(R.shape)
        R = (R + 1) / 2
        # import pdb; pdb.set_trace()
        Full[:, :, i] = R
        # file_out =  join(outpath, f"WECS_R{alpha}_{win}_{name[i]}_080117_221217.tiff")
        # array2raster(R[:, :, np.newaxis], geo, file_out, gdal_driver="GTiff")

    file_out = join(outpath, f"WECS_R{alpha}_{win}_4C_080117_221217.tiff")
    array2raster(Full, geo, file_out, gdal_driver="GTiff")
    del Full, R, d_matrices, d_vector, x_images, images, mean_image
    return 1


if __name__ == "__main__":
    path = "../dataset_E1/*.tiff"

    folder = str(time.time()).split(".")[0]
    outpath = f"../dataset_E2/{folder}"
    exist_create_folder(outpath)
    alphas = [0.25, 0.5, 0.75, 0.875]
    wins = [
        1,
        2,
        5,
        10,
    ]
    run_wecs(path, outpath, alphas, wins)
