import numpy as np
from osgeo import gdal
from os.path import dirname, abspath, join, basename, exists
from os import makedirs
import glob
from tqdm import tqdm
import subprocess


def load_data(file_name, gdal_driver="GTiff"):
    """
    Load a raster file as a numpy array and get the geodata.

    Parameters
    ----------
    file_name : str
        path to raster file
    gdal_driver : str, optional
        gdal driver to use. The default is "GTiff".

    Returns
    -------
    image_array : numpy array

    (geotransform, projection) : tuple
    """
    driver = gdal.GetDriverByName(gdal_driver)  ## http://www.gdal.org/formats_list.html
    driver.Register()

    inDs = gdal.Open(file_name, gdal.GA_ReadOnly)

    if inDs is None:
        print("Couldn't open this file: %s" % (file_name))
    else:
        pass
    # Extract some info form the inDs
    geotransform = inDs.GetGeoTransform()
    projection = inDs.GetProjection()

    # Get the data as a numpy array
    cols = inDs.RasterXSize
    rows = inDs.RasterYSize

    channel = inDs.RasterCount
    image_array = np.zeros((rows, cols, channel), dtype=np.float32)
    for i in range(channel):
        data_array = inDs.GetRasterBand(i + 1).ReadAsArray(0, 0, cols, rows)
        image_array[:, :, i] = data_array
    inDs = None
    return image_array, (geotransform, projection)


def array2raster(data_array, geodata, file_out, gdal_driver="GTiff"):
    """Write a numpy array to a raster file.

    Parameters
    ----------
    data_array : numpy array
        array to write to raster
    geodata : tuple
        geotransform and projection of original raster
    file_out : str
        output file path
    gdal_driver : str, optional
        gdal driver to use. The default is "GTiff".

    Returns
    -------
    None.
    """

    if not exists(dirname(file_out)):
        print("Your output directory doesn't exist - please create it")
        print("No further processing will take place.")
    else:
        post = geodata[0][1]
        original_geotransform, projection = geodata

        rows, cols, bands = data_array.shape
        # adapt number of bands to input data

        # Set the gedal driver to use
        driver = gdal.GetDriverByName(gdal_driver)
        driver.Register()

        # Creates a new raster data source
        outDs = driver.Create(file_out, cols, rows, bands, gdal.GDT_Float32)

        # Write metadata
        originX = original_geotransform[0]
        originY = original_geotransform[3]

        outDs.SetGeoTransform([originX, post, 0.0, originY, 0.0, -post])
        outDs.SetProjection(projection)

        # Write raster datasets
        for i in range(bands):
            outBand = outDs.GetRasterBand(i + 1)
            outBand.SetNoDataValue(-999)
            outBand.WriteArray(data_array[:, :, i])

        print("Output saved: %s" % file_out)


def gdal_clip_shp_raster(inraster, inshape, outraster, country_name):
    """Clip raster with shapefile

    Parameters
    ----------
    inraster : str
        input raster path
    inshape : str
        input shapefile path
    outraster : str
        output raster path
    country_name : str
        id in the shapefile

    Returns
    -------
    None.
    """
    subprocess.call(
        [
            "gdalwarp",
            "-of",
            "Gtiff",
            "-dstnodata",
            "value -999",
            "-ot",
            "Float32",
            inraster,
            outraster,
            "-cutline",
            inshape,
            "-crop_to_cutline",
            "-cwhere",
            f"id='{country_name}'",
        ]
    )


if __name__ == "__main__":
    ################ 1. Clip raster with shapefile ################
    imag_path = "../reunion/s1_process/*.TIF"
    oudir = "../reunion/PC_R_0"  # 1111_P_crop"
    makedirs(oudir, exist_ok=True)
    inshape = glob.glob("../reunion/shp/*.shp")[0]
    ra, gdt = load_data("../reunion/shp/raster_shp.tif")
    ra = ra[:, :, 0]
    if True:
        print(inshape)
        for i in tqdm(glob.glob(imag_path)):
            name = basename(i)[:29] + "_CROP_.tiff"
            outraster = join(oudir, name)
            gdal_clip_shp_raster(i, inshape, outraster, 1)

    ################ 2. Combine & log10 ################
    if True:
        imag_path = join(oudir, "*.tiff")
        oudir = "../reunion/PC_R_1"
        makedirs(oudir, exist_ok=True)

        for i in tqdm(glob.glob(imag_path)):
            vv_img, gdata = load_data(i, gdal_driver="GTiff")
            vv = vv_img[:, :, 0]
            vh = vv_img[:, :, 1]
            vv = np.where(vv <= 0, np.nan, vv)
            vh = np.where(vh <= 0, np.nan, vh)
            vdb = 10 * np.log10(vv)
            hdb = 10 * np.log10(vh)
            dif = vdb - hdb
            new_array = np.dstack((vdb, hdb, dif))

            name = basename(i)[:-5] + "3C.tiff"
            file_name = join(oudir, name)

            array2raster(new_array, gdata, file_name)

    ######################################################
