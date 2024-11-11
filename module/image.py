from osgeo import gdal, ogr
import numpy as np
import os


class ImageProcess:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = gdal.Open(self.filepath, gdal.GA_ReadOnly)
        self.info = []
        self.img_data = None
        self.data_8bit = None

    def read_img_info(self):
        # 获取波段、宽、高
        img_bands = self.dataset.RasterCount
        img_width = self.dataset.RasterXSize
        img_height = self.dataset.RasterYSize
        # 获取仿射矩阵、投影
        img_geotrans = self.dataset.GetGeoTransform()
        img_proj = self.dataset.GetProjection()
        # 获取NoData值
        img_nodata = self.dataset.GetRasterBand(1).GetNoDataValue()
        self.info = [img_bands, img_width, img_height, img_geotrans, img_proj,img_nodata]
        return self.info

    def read_img_data(self):
        self.img_data = self.dataset.ReadAsArray(0, 0, self.info[1], self.info[2])
        return self.img_data

    # 影像写入文件
    @staticmethod
    def write_img(filename: str, img_data: np.array, **kwargs):
        # 判断栅格数据的数据类型
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(img_data.shape) >= 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        outdataset = driver.Create(filename, img_width, img_height, img_bands, datatype)
        # 写入仿射变换参数
        if 'img_geotrans' in kwargs:
            outdataset.SetGeoTransform(kwargs['img_geotrans'])
        # 写入投影
        if 'img_proj' in kwargs:
            outdataset.SetProjection(kwargs['img_proj'])
        # 写入文件
        if img_bands == 1:
            outdataset.GetRasterBand(1).WriteArray(img_data)  # 写入数组数据
        else:
            for i in range(img_bands):
                outdataset.GetRasterBand(i + 1).WriteArray(img_data[i])

        del outdataset


def read_multi_bands(image_path):
    """
    读取多波段文件
    :param image_path: 多波段文件路径
    :return: 影像对象，影像元信息，影像矩阵
    """
    # 影像读取
    image = ImageProcess(filepath=image_path)
    # 读取影像元信息
    image_info = image.read_img_info()
    # print(f"多波段影像元信息：{image_info}")
    # 读取影像矩阵
    image_data = image.read_img_data()
    print(f"多波段矩阵大小：{image_data.shape}")
    return image, image_info, image_data


def read_single_band(band_path):
    """
    读取单波段文件
    :param band_path: 单波段文件路径
    :return: 影像对象，影像元信息，影像矩阵
    """
    # 影像读取
    band = ImageProcess(filepath=band_path)
    # 读取影像元信息
    band_info = band.read_img_info()
    # print(f"单波段影像元信息：{band_info}")
    # 读取影像矩阵
    band_data = band.read_img_data()
    print(f"单波段矩阵大小：{band_data.shape}")
    return band, band_info, band_data

def resample_tif(input_tif, reference_tif, output_tif):
    """
    使用参考影像的像元大小和影像范围对输入影像进行重采样
    :param input_tif: 需要重采样的影像路径
    :param reference_tif: 参考影像路径
    :param output_tif: 重采样后的影像输出路径
    """
    # 打开输入影像和参考影像
    input_ds = gdal.Open(input_tif)
    reference_ds = gdal.Open(reference_tif)

    # 获取参考影像的地理变换和投影信息
    reference_geotransform = reference_ds.GetGeoTransform()
    reference_projection = reference_ds.GetProjection()
    reference_width = reference_ds.RasterXSize
    reference_height = reference_ds.RasterYSize

    # 创建重采样后的影像
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_tif, reference_width, reference_height, input_ds.RasterCount, gdal.GDT_Float32)
    output_ds.SetGeoTransform(reference_geotransform)
    output_ds.SetProjection(reference_projection)

    # 使用GDAL重采样算法进行重采样,使用最近邻插值
    gdal.ReprojectImage(input_ds, output_ds, input_ds.GetProjection(), reference_projection, gdal.GRA_NearestNeighbour)

    # 关闭数据集
    input_ds = None
    reference_ds = None
    output_ds = None

    print(f"重采样完成，输出影像保存为：{output_tif}")

def clip_tif_by_shapefile(input_tif, shapefile, output_tif):
    """
    根据shapefile的范围裁剪输入影像
    :param input_tif: 需要裁剪的影像路径
    :param shapefile: 用于裁剪的shapefile路径
    :param output_tif: 裁剪后的影像输出路径
    """
    # 打开输入影像
    input_ds = gdal.Open(input_tif)
    if input_ds is None:
        raise FileNotFoundError(f"无法打开输入影像：{input_tif}")

    # 打开shapefile
    shapefile_ds = ogr.Open(shapefile)
    if shapefile_ds is None:
        raise FileNotFoundError(f"无法打开shapefile：{shapefile}")

    shapefile_layer = shapefile_ds.GetLayer()

    # 获取shapefile的范围
    x_min, x_max, y_min, y_max = shapefile_layer.GetExtent()

    # 使用GDAL Warp函数进行裁剪
    options = gdal.WarpOptions(
        format='GTiff',
        cutlineDSName=shapefile,
        cropToCutline=True,
        dstNodata=0
    )
    gdal.Warp(output_tif, input_ds, options=options)

    # 关闭数据集
    input_ds = None
    shapefile_ds = None

    print(f"裁剪完成，输出影像保存为：{output_tif}")

# 示例调用
# input_tif = r'E:\2024Work\陆表赛道-基于高分卫星的农作物精细识别技术\code\HSI_SSFTT-main\data\train\GF2_train_label\GF5_train_label.tif'
# shapefile = r'E:\2024Work\陆表赛道-基于高分卫星的农作物精细识别技术\code\HSI_SSFTT-main\data\GF5_RANGE.shp'
# output_tif = r'E:\2024Work\陆表赛道-基于高分卫星的农作物精细识别技术\code\HSI_SSFTT-main\data\train\GF2_train_label\GF2_train_label_clipped.tif'
# clip_tif_by_shapefile(input_tif, shapefile, output_tif)
# 示例调用
# input_tif = r'E:\2024Work\陆表赛道-基于高分卫星的农作物精细识别技术\code\HSI_SSFTT-main\data\train\GF2_train_label\GF2_train_label_clipped.tif'
# reference_tif = r'E:\2024Work\陆表赛道-基于高分卫星的农作物精细识别技术\code\HSI_SSFTT-main\data\train\GF2_train_image\GF5_train_image.tif'
# output_tif = r'E:\2024Work\陆表赛道-基于高分卫星的农作物精细识别技术\code\HSI_SSFTT-main\data\train\GF2_train_label\GF5_train_label.tif'
# resample_tif(input_tif, reference_tif, output_tif)