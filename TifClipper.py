'''
本代码用于从原始城市label图像、image图像、VVH图像、多季节图像分割数据集

Tips：
1. 如果需要对img进行超分，不需要将DN值归一化再×255，输出原始值即可
2. label的Nodata值经过一次训练之后，认为设置为0更方便训练（也可以在网络设置NoData值）
3. 相对路径os库可以读取/开头的路径，但是gdal库不能读取，要加.   比如'./Data/label'而不是'/Data/label'
author: LyuSiqing
'''

from osgeo import gdal
import numpy as np
import math
import time
from tqdm import tqdm
import os
from tools import *

'''--------------------------Configs-------------------------'''
Nodata_lab = 0
Nodata_img = -3.40282346639e+038



def clip_1dim(dataset_GT, outpath='GT.tif', offset_x=0,
                offset_y=0, block_xsize=128, block_ysize=128, Nodata = 0):

    in_band1 = dataset_GT.GetRasterBand(1)
    out_band1 = in_band1.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    # if np.all(out_band1 == Nodata):
    #     return False
    # sum = 0
    # for da in out_band1:
    #     for i in da:
    #         if i == 1:
    #             sum += 1
    # if sum < 5:
    #     return False
    gtif_driver = gdal.GetDriverByName("GTiff")
    # 创建切出来的要存的文件（最后一个参数为数据类型，跟原文件一致）
    out_ds = gtif_driver.Create(outpath, block_xsize, block_ysize, 1, in_band1.DataType)
    # print("create new tif file succeed")
    geomat = dataset_GT.GetGeoTransform()
    # if geomat:
    #     print(geomat)
    #     print("Origin = ({}, {})".format(geomat[0], geomat[3]))
    #     print("Pixel Size = ({}, {})".format(geomat[1], geomat[5]))

    # 读取原图仿射变换参数值
    top_left_x = geomat[0]  # 左上角x坐标
    w_e_pixel_resolution = geomat[1]  # 东西方向像素分辨率
    top_left_y = geomat[3]  # 左上角y坐标
    n_s_pixel_resolution = geomat[5]  # 南北方向像素分辨率

    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x + offset_x * w_e_pixel_resolution
    top_left_y = top_left_y + offset_y * n_s_pixel_resolution

    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, geomat[1], geomat[2], top_left_y, geomat[4], geomat[5])

    # 设置裁剪出来图的原点坐标
    out_ds.SetGeoTransform(dst_transform)

    # 设置SRS属性（投影信息）
    out_ds.SetProjection(dataset_GT.GetProjection())
    out_ds.GetRasterBand(1).SetNoDataValue(Nodata_lab)
    # 写入目标文件
    # out_band1 = out_band1 * 3
    out_ds.GetRasterBand(1).WriteArray(out_band1)

    # 将缓存写入磁盘
    out_ds.FlushCache()
    # print("FlushCache succeed")
    del out_ds
    return True


def cliptif_4dim(dataset, outpath='Clip.tif', offset_x=0, offset_y=0,
                 block_xsize=128, block_ysize=128, Nodata = 0):
    in_band1 = dataset.GetRasterBand(1)
    in_band2 = dataset.GetRasterBand(2)
    in_band3 = dataset.GetRasterBand(3)
    in_band4 = dataset.GetRasterBand(4)

    out_band1 = in_band1.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    out_band2 = in_band2.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    out_band3 = in_band3.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)
    out_band4 = in_band4.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)

    # sum = 0
    # for da in out_band1:
    #     for i in da:
    #         if i >= 0 :
    #             sum += 1
    # if sum < 0.80 * block_xsize * block_ysize:
    #     return False

    gtif_driver = gdal.GetDriverByName("GTiff")
    # 创建切出来的要存的文件（最后一个参数为数据类型，跟原文件一致）
    out_ds = gtif_driver.Create(outpath, block_xsize, block_ysize, 4, in_band1.DataType)
    # print("create new tif file succeed")
    geomat = dataset.GetGeoTransform()
    # if geomat:
    #     print(geomat)
    #     print("Origin = ({}, {})".format(geomat[0], geomat[3]))
    #     print("Pixel Size = ({}, {})".format(geomat[1], geomat[5]))

    # 读取原图仿射变换参数值
    top_left_x = geomat[0]  # 左上角x坐标
    w_e_pixel_resolution = geomat[1]  # 东西方向像素分辨率
    top_left_y = geomat[3]  # 左上角y坐标
    n_s_pixel_resolution = geomat[5]  # 南北方向像素分辨率

    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x + offset_x * w_e_pixel_resolution
    top_left_y = top_left_y + offset_y * n_s_pixel_resolution

    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, geomat[1], geomat[2], top_left_y, geomat[4], geomat[5])

    # 设置裁剪出来图的原点坐标
    out_ds.SetGeoTransform(dst_transform)

    # 设置SRS属性（投影信息）
    out_ds.SetProjection(dataset.GetProjection())
    out_ds.GetRasterBand(1).SetNoDataValue(Nodata)
    out_ds.GetRasterBand(2).SetNoDataValue(Nodata)
    out_ds.GetRasterBand(3).SetNoDataValue(Nodata)
    out_ds.GetRasterBand(4).SetNoDataValue(Nodata)
    # 写入目标文件  1234 = B G R N
    out_ds.GetRasterBand(1).WriteArray(out_band1)
    out_ds.GetRasterBand(2).WriteArray(out_band2)
    out_ds.GetRasterBand(3).WriteArray(out_band3)
    out_ds.GetRasterBand(4).WriteArray(out_band4)

    # 将缓存写入磁盘
    out_ds.FlushCache()
    # print("FlushCache succeed")
    del out_ds
    return True





def cal_gaps(dataset_a, dataset_b):
    '''
    to calculate the gap between two tifs
    :param dataset_a: this file is the smaller one
    :param dataset_b: this file is the bigger one, should be the reference
    :return: (x,y) for adjusting the clipper position
    '''
    geomat = dataset_a.GetGeoTransform()
    geomat_b = dataset_b.GetGeoTransform()
    x1 = geomat_b[0]
    y1 = geomat_b[3]
    dTemp = geomat[1] * geomat[5] - geomat[2] * geomat[4]
    x_gap = (geomat[5] * (x1 - geomat[0]) - geomat[2] * (y1 - geomat[3])) / dTemp
    y_gap = (geomat[1] * (y1 - geomat[3]) - geomat[4] * (x1 - geomat[3])) / dTemp

    # print("dataset_a: x0:{0}, y0:{1}\ndataset_b: x0:{2}, y0:{3}".format(geomat[0], geomat[3], geomat_b[0], geomat_b[3]))
    return x_gap, y_gap



