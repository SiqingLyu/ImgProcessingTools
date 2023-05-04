from tools import make_dir, file_name_tif
from osgeo import gdal
import numpy as np
import os
import tifffile as tif


def asign(from_img, to_img, out_path):
    out_band0 = tif.imread(from_img)
    out_band0 = np.round(out_band0)

    dataset = gdal.Open(to_img, gdal.GA_Update)
    dataset.FlushCache()
    in_band1 = dataset.GetRasterBand(1)
    out_band1 = in_band1.ReadAsArray()
    gtif_driver = gdal.GetDriverByName("GTiff")
    # 创建切出来的要存的文件（最后一个参数为数据类型，跟原文件一致）
    out_ds = gtif_driver.Create(out_path, out_band1.shape[1], out_band1.shape[0], 1, in_band1.DataType)
    # print("create new tif file succeed")
    geomat = dataset.GetGeoTransform()
    top_left_x = geomat[0]  # 左上角x坐标
    w_e_pixel_resolution = geomat[1]  # 东西方向像素分辨率
    top_left_y = geomat[3]  # 左上角y坐标
    n_s_pixel_resolution = geomat[5]  # 南北方向像素分辨率

    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x
    top_left_y = top_left_y

    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, geomat[1], geomat[2], top_left_y, geomat[4], geomat[5])

    # 设置裁剪出来图的原点坐标
    out_ds.SetGeoTransform(dst_transform)

    # 设置SRS属性（投影信息）
    out_ds.SetProjection(dataset.GetProjection())
    out_ds.GetRasterBand(1).SetNoDataValue(0)
    # 写入目标文件
    # out_band1 = out_band1 * 3
    out_ds.GetRasterBand(1).WriteArray(out_band0)
    # 将缓存写入磁盘
    out_ds.FlushCache()
    # print("FlushCache succeed")
    del out_ds


def main(from_dir = r'', to_dir = r'', save_dir = r''):
    from_img_paths, from_img_names = file_name_tif(from_dir)
    to_img_paths, to_img_names = file_name_tif(to_dir)
    for ii in range(len(from_img_paths)):
        from_img_name = from_img_names[ii]
        # print(from_img_name)
        if from_img_name in to_img_names:
            print(f'processing: {from_img_name}')
            asign(from_img=from_img_paths[ii], to_img=os.path.join(to_dir, from_img_name+'.tif'), out_path=os.path.join(save_dir, from_img_name+'.tif'))


if __name__ == '__main__':
    # 要赋值地理坐标的文件所在位置
    from_dir = r'D:\experiment\results\Shanghai_ours_UNet_M3Net_footprint_SSN_296_V2\pred'
    # 包含目标地理坐标的文件所在位置，这两个文件名必须一样
    to_dir = r'D:\PythonProjects\DataProcess\Data\SHANGHAI_OURS\label'
    # 结果存储位置
    save_dir = make_dir(r'D:\experiment\results\Shanghai_ours_UNet_M3Net_footprint_SSN_296_V2\Assigned')
    main(from_dir, to_dir, save_dir)