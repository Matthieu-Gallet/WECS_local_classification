import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap
import skimage
from PIL import Image
import matplotlib.colors
import matplotlib.patches
import cv2 
from tkinter import messagebox
import os


def scale(im):
    scaled = (im - np.nanmin(im))/(np.nanmax(im)-np.nanmin(im))
    return im

def calculate_A(images):
    n = len(images)
    h,w = np.shape(images[0])
    agg = np.zeros((h,w))

    for i in range(1, n):
        agg = agg + abs(images[i]-images[i-1])

    return agg

def change_map_otsu(R):
    R_grayscale = np.array(Image.fromarray(np.uint8(R * 255) , 'L'))
    o = skimage.filters.threshold_otsu(R_grayscale)
    binary_mask = R_grayscale > o

    return binary_mask

def read_time_series():
    images = []

    #Reading images
    directory = 'code/IMAGE_SERIES/'
    for file in os.listdir(directory):
        im = gdal.Open(os.path.join(directory, file))
        imarray = im.ReadAsArray().astype(float)
        im_comb = (np.nansum(np.square(np.dstack(scale(imarray))), axis = 2))
        images.append(im_comb)
    
    images = np.asarray(images)
    return images

def change_map_kmeans(R, k):
    flat = np.float32(R.flatten())
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0002)

    _, labels, (centers) = cv2.kmeans(flat, k, None, criteria, 10,  cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.flatten()
    res = centers[labels]
    change = np.nanmax(centers) #high correlation values represent change
    binary_mask = np.where((res == change), 1, 0).reshape(np.shape(R))

    return binary_mask

def show_results(R, otsu, k3, k4):
    cmap = matplotlib.colors.ListedColormap(['lightgreen', 'red'])
    fig = plt.figure()

    fig.add_subplot(2,2,1)
    plt.imshow(R, cmap='rainbow')
    plt.colorbar()
    plt.title("Aggregation Matrix")
    plt.axis('off')

    fig.add_subplot(2,2,2)
    plt.imshow(otsu, cmap=cmap)
    plt.title("Otsu thresholding")
    plt.axis('off')

    fig.add_subplot(2,2,3)
    plt.imshow(k3, cmap=cmap)
    plt.title("Kmeans with K=3")
    plt.axis('off')

    fig.add_subplot(2,2,4)
    plt.imshow(k4, cmap=cmap)
    plt.title("Kmeans with K=4")
    plt.axis('off')

    plt.show()

def show_images(images):
    p = plt.figure()

    p.add_subplot(1,2,1)
    plt.imshow(images[0], cmap = 'gray')
    plt.title('First image')
    plt.axis('off')

    p.add_subplot(1,2,2)
    plt.imshow(images[len(images)-1], cmap = 'gray')
    plt.title('Last image')
    plt.axis('off')

    plt.annotate('While this window is open, the method is frozen. \nClose to continue execution.',
            xy = (1.0, -0.2),
            xycoords='axes fraction',
            ha='right',
            va="center",
            fontsize=10)

    plt.show() 

def run_taad():
    images = read_time_series()
    show_images(images)

    #aggregation matrix
    A = calculate_A(images)
    A = A/np.nanmax(A)

    #Binary maps
    cm_otsu = change_map_otsu(A)
    cm_kmeans_3 = change_map_kmeans(A, 3)
    cm_kmeans_4 = change_map_kmeans(A, 4)

    show_results(A, cm_otsu, cm_kmeans_3, cm_kmeans_4)

    cmap = matplotlib.colors.ListedColormap(['lightgreen', 'red'])
    matplotlib.image.imsave('code/CHANGE_MAPS/TAAD/change_map_otsu.png', np.asarray(cm_otsu), cmap=cmap)
    matplotlib.image.imsave('code/CHANGE_MAPS/TAAD/change_map_kmeans_3.png', np.asarray(cm_kmeans_3), cmap=cmap)
    matplotlib.image.imsave('code/CHANGE_MAPS/TAAD/change_map_kmeans_4.png', np.asarray(cm_kmeans_4), cmap=cmap)
    matplotlib.image.imsave('code/CHANGE_MAPS/TAAD/aggregation_matrix_A.png', np.asarray(A), cmap='rainbow')

    messagebox.showinfo("MESSAGE", "The change maps and correlation matrix have been saved in the folder \'code/CHANGE_MAPS/WECS\'. Make sure to transport it to your own storage because the files will be lost in the next execution of the method.")

