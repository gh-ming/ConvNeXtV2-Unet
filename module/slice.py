import math
import random
import numpy as np
from alive_progress import alive_bar
from module.image import *
def check_dir(dir_path):
    """
    检查文件夹是否存在，不存在则创建
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def cal_single_band_slice(single_band_data, slice_size=1000):
    """
    计算单波段的格网裁剪四角点
    :param single_band_data:单波段原始数据
    :param slice_size: 裁剪大小
    :return: 嵌套列表，每一个块的四角行列号
    """
    single_band_size = single_band_data.shape
    row_num = math.ceil(single_band_size[0] / slice_size)  # 向上取整
    col_num = math.ceil(single_band_size[1] / slice_size)  # 向上取整
    print(f"行列数：{single_band_size}，行分割数量：{row_num}，列分割数量：{col_num}")
    slice_index = []
    for i in range(row_num):
        for j in range(col_num):
            row_min = i * slice_size
            row_max = (i + 1) * slice_size
            if (i + 1) * slice_size > single_band_size[0]:
                row_max = single_band_size[0]
            col_min = j * slice_size
            col_max = (j + 1) * slice_size
            if (j + 1) * slice_size > single_band_size[1]:
                col_max = single_band_size[1]
            slice_index.append([row_min, row_max, col_min, col_max])
    return slice_index


def single_band_slice(single_band_data, index=[0, 1000, 0, 1000], slice_size=1000, edge_fill=False):
    """
    依据四角坐标，切分单波段影像
    :param single_band_data:原始矩阵数据
    :param index: 四角坐标
    :param slice_size: 分块大小
    :param edge_fill: 是否进行边缘填充
    :return: 切分好的单波段矩阵
    """
    if edge_fill:
        if (index[1] - index[0] != slice_size) or (index[3] - index[2] != slice_size):
            result = np.empty(shape=(slice_size, slice_size))
            new_row_min = index[0] % slice_size
            new_row_max = new_row_min + (index[1] - index[0])
            new_col_min = index[2] % slice_size
            new_col_max = new_col_min + (index[3] - index[2])
            result[new_row_min:new_row_max, new_col_min:new_col_max] = single_band_data[index[0]:index[1],
                                                                       index[2]:index[3]]
        else:
            result = single_band_data[index[0]:index[1], index[2]:index[3]]
    else:
        result = single_band_data[index[0]:index[1], index[2]:index[3]]
    return result.astype(single_band_data.dtype)


def multi_bands_slice(multi_bands_data, index=[0, 1000, 0, 1000], slice_size=1000, edge_fill=False):
    """
    依据四角坐标，切分多波段影像
    :param multi_bands_data: 原始多波段矩阵
    :param index: 四角坐标
    :param slice_size: 分块大小
    :param edge_fill: 是否进行边缘填充
    :return: 切分好的多波段矩阵
    """
    if edge_fill:
        if (index[1] - index[0] != slice_size) or (index[3] - index[2] != slice_size):
            result = np.empty(shape=(multi_bands_data.shape[0], slice_size, slice_size))
            new_row_min = index[0] % slice_size
            new_row_max = new_row_min + (index[1] - index[0])
            new_col_min = index[2] % slice_size
            new_col_max = new_col_min + (index[3] - index[2])
            result[:, new_row_min:new_row_max, new_col_min:new_col_max] = multi_bands_data[:, index[0]:index[1],
                                                                          index[2]:index[3]]
        else:
            result = multi_bands_data[:, index[0]:index[1], index[2]:index[3]]
    else:
        result = multi_bands_data[:, index[0]:index[1], index[2]:index[3]]
    return result.astype(multi_bands_data.dtype)


def slice_conbine(slice_all, slice_index):
    """
    将分块矩阵进行合并
    :param slice_all: 所有的分块矩阵列表
    :param slice_index: 分块的四角坐标
    :return: 合并的矩阵
    """
    combine_data = np.zeros(shape=(slice_index[-1][1], slice_index[-1][3]))
    # print(combine_data.shape)
    for i, slice_element in enumerate(slice_index):
        combine_data[slice_element[0]:slice_element[1], slice_element[2]:slice_element[3]] = slice_all[i]
    return combine_data


def coordtransf(Xpixel, Ypixel, GeoTransform):
    """
    像素坐标和地理坐标仿射变换
    :param Xpixel: 左上角行号
    :param Ypixel: 左上角列号
    :param GeoTransform: 原始仿射矩阵
    :return: 新的仿射矩阵
    """
    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2];
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5];
    slice_geotrans = (XGeo, GeoTransform[1], GeoTransform[2], YGeo, GeoTransform[4], GeoTransform[5])
    return slice_geotrans


def multi_bands_grid_slice(image_path, image_slice_dir, slice_size, edge_fill=False):
    """
    多波段格网裁剪
    :param image_path: 原始多波段影像
    :param image_slice_dir: 裁剪保存文件夹
    :param slice_size: 裁剪大小
    :return:
    """
    image, image_info, image_data = read_multi_bands(image_path)
    # 计算分块的四角行列号
    slice_index = cal_single_band_slice(image_data[0, :, :], slice_size=slice_size, )
    # 执行裁剪
    with alive_bar(len(slice_index), force_tty=True) as bar:
        for i, slice_element in enumerate(slice_index):
            slice_data = multi_bands_slice(image_data, index=slice_element, slice_size=slice_size,
                                           edge_fill=edge_fill)  # 裁剪多波段影像
            slice_geotrans = coordtransf(slice_element[2], slice_element[0], image_info[3])  # 转换仿射坐标
            image.write_img(image_slice_dir + r'\multi_grid_slice_' + str(i) + '.tif', slice_data,
                            img_geotrans=slice_geotrans, img_proj=image_info[4])  # 写入文件
            bar()
        print('多波段格网裁剪完成')


def single_band_grid_slice(band_path, band_slice_dir, slice_size, edge_fill=False):
    """
    单波段格网裁剪
    :param band_path: 原始单波段影像
    :param band_slice_dir: 裁剪保存文件夹
    :param slice_size: 裁剪大小
    :return:
    """
    band, band_info, band_data = read_single_band(band_path)
    # 计算分块的四角行列号
    slice_index = cal_single_band_slice(band_data, slice_size=slice_size)
    # 执行裁剪
    with alive_bar(len(slice_index), force_tty=True) as bar:
        for i, slice_element in enumerate(slice_index):
            slice_data = single_band_slice(band_data, index=slice_element, slice_size=slice_size,
                                           edge_fill=edge_fill)  # 裁剪单波段影像
            slice_geotrans = coordtransf(slice_element[2], slice_element[0], band_info[3])  # 转换仿射坐标
            band.write_img(band_slice_dir + r'\single_grid_slice_' + str(i) + '.tif', slice_data,
                           img_geotrans=slice_geotrans, img_proj=band_info[4])  # 写入文件
            bar()
        print('单波段格网裁剪完成')


def multi_bands_rand_slice(image_path, image_slice_dir, slice_size, slice_count):
    """
    多波段随机裁剪
    :param image_path: 原始多波段影像
    :param image_slice_dir: 裁剪保存文件夹
    :param slice_size: 裁剪大小
    :param slice_count: 裁剪数量
    :return:
    """
    image, image_info, image_data = read_multi_bands(image_path)
    # 生成随机起始点
    randx = [random.randint(0, image_info[2] - slice_size - 1) for i in range(slice_count)]
    randy = [random.randint(0, image_info[1] - slice_size - 1) for j in range(slice_count)]
    randx1 = np.add(randx, slice_size).tolist()
    randy1 = np.add(randy, slice_size).tolist()
    rand_index = [[randx[k], randx1[k], randy[k], randy1[k]] for k in range(slice_count)]
    # 进行裁剪
    with alive_bar(len(rand_index), force_tty=True) as bar:
        for i, slice_element in enumerate(rand_index):
            slice_data = multi_bands_slice(image_data, index=slice_element, slice_size=slice_size)  # 裁剪多波段影像
            slice_geotrans = coordtransf(slice_element[2], slice_element[0], image_info[3])  # 转换仿射坐标
            image.write_img(image_slice_dir + r'\multi_rand_slice_' + str(i) + '.tif', slice_data,
                            img_geotrans=slice_geotrans, img_proj=image_info[4])  # 写入文件
            bar()
        print('多波段随机裁剪完成')


def single_band_rand_slice(band_path, band_slice_dir, slice_size, slice_count):
    """
    单波段随机裁剪
    :param band_path: 原始单波段影像
    :param band_slice_dir: 裁剪保存文件夹
    :param slice_size: 裁剪大小
    :param slice_count: 裁剪数量
    :return:
    """
    band, band_info, band_data = read_single_band(band_path)
    # 生成随机起始点
    randx = [random.randint(0, band_info[2] - slice_size - 1) for i in range(slice_count)]
    randy = [random.randint(0, band_info[1] - slice_size - 1) for j in range(slice_count)]
    randx1 = np.add(randx, slice_size).tolist()
    randy1 = np.add(randy, slice_size).tolist()
    rand_index = [[randx[k], randx1[k], randy[k], randy1[k]] for k in range(slice_count)]
    # 进行裁剪
    with alive_bar(len(rand_index), force_tty=True) as bar:
        for i, slice_element in enumerate(rand_index):
            slice_data = single_band_slice(band_data, index=slice_element, slice_size=slice_size)  # 裁剪单波段影像
            slice_geotrans = coordtransf(slice_element[2], slice_element[0], band_info[3])  # 转换仿射坐标
            band.write_img(band_slice_dir + r'\single_rand_slice_' + str(i) + '.tif', slice_data,
                           img_geotrans=slice_geotrans, img_proj=band_info[4])  # 写入文件
            bar()
        print('单波段随机裁剪完成')


def deeplr_grid_slice(image_path, band_path, image_slice_dir, band_slice_dir, slice_size, edge_fill=False,nodata_ignore=False):
    """
    制作深度学习样本-格网裁剪：同时裁剪多波段、单波段影像
    :param image_path: 原始image影像
    :param band_path: 原始label影像
    :param image_slice_dir: image裁剪保存文件夹
    :param band_slice_dir: label裁剪保存文件夹
    :param slice_size: 裁剪大小
    :return:
    """
    check_dir(image_slice_dir)
    check_dir(band_slice_dir)
    image, image_info, image_data = read_multi_bands(image_path)
    band, band_info, band_data = read_single_band(band_path)

    # if image_data.shape[1:] != band_data.shape:
    #     raise ValueError("The size of the image and label do not match.")
    
    # Padding the band_data matrix with an extra row and column filled with zeros
    band_data = np.pad(band_data, ((0, 1), (0, 1)), mode='constant', constant_values=0)

    # 计算分块的四角行列号
    slice_index = cal_single_band_slice(image_data[0, :, :], slice_size=slice_size)
    # 执行裁剪
    with alive_bar(len(slice_index), force_tty=True) as bar:
        for i, slice_element in enumerate(slice_index):
            slice_data = multi_bands_slice(image_data, index=slice_element, slice_size=slice_size,
                                           edge_fill=edge_fill)  # 裁剪多波段影像
            # 忽略nodata值
            if nodata_ignore:
                if np.all(slice_data == image_info[5]):
                    continue
                if  np.all(slice_data == 0):
                    continue
            slice_geotrans = coordtransf(slice_element[2], slice_element[0], image_info[3])  # 转换仿射坐标
            img_path = os.path.join(image_slice_dir, str(i) + '.tif')
            label_path = os.path.join(band_slice_dir, str(i) + '.tif')
            image.write_img(img_path, slice_data,
                            img_geotrans=slice_geotrans, img_proj=image_info[4])  # 写入文件

            slice_band = single_band_slice(band_data, index=slice_element, slice_size=slice_size,
                                           edge_fill=edge_fill)  # 裁剪单波段影像
            band.write_img(label_path, slice_band,
                           img_geotrans=slice_geotrans, img_proj=band_info[4])  # 写入文件
            bar()
        print('深度学习样本-格网裁剪完成')


def deeplr_rand_slice(image_path, band_path, image_slice_dir, band_slice_dir, slice_size, slice_count):
    """
    制作深度学习样本-随机裁剪：同时裁剪多波段、单波段影像
    :param image_path: 原始image影像
    :param band_path: 原始label影像
    :param image_slice_dir: image裁剪保存文件夹
    :param band_slice_dir: label裁剪保存文件夹
    :param slice_size: 裁剪大小
    :param slice_count: 裁剪数量
    :return:
    """
    image, image_info, image_data = read_multi_bands(image_path)
    band, band_info, band_data = read_single_band(band_path)
    # 生成随机起始点
    randx = [random.randint(0, image_info[2] - slice_size - 1) for i in range(slice_count)]
    randy = [random.randint(0, image_info[1] - slice_size - 1) for j in range(slice_count)]
    randx1 = np.add(randx, slice_size).tolist()
    randy1 = np.add(randy, slice_size).tolist()
    rand_index = [[randx[k], randx1[k], randy[k], randy1[k]] for k in range(slice_count)]
    # 执行裁剪
    with alive_bar(len(rand_index), force_tty=True) as bar:
        for i, slice_element in enumerate(rand_index):
            slice_data = multi_bands_slice(image_data, index=slice_element, slice_size=slice_size)  # 裁剪多波段影像
            slice_geotrans = coordtransf(slice_element[2], slice_element[0], image_info[3])  # 转换仿射坐标
            image.write_img(image_slice_dir + r'\multi_rand_slice_' + str(i) + '.tif', slice_data,
                            img_geotrans=slice_geotrans, img_proj=image_info[4])  # 写入文件

            slice_band = single_band_slice(band_data, index=slice_element, slice_size=slice_size)  # 裁剪单波段影像
            band.write_img(band_slice_dir + r'\single_rand_slice_' + str(i) + '.tif', slice_band,
                           img_geotrans=slice_geotrans, img_proj=band_info[4])  # 写入文件
            bar()
        print('深度学习样本-随机裁剪完成')
