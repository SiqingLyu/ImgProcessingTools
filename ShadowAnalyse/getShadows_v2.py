
"""get the shadow proportion form images
   of remote sensing"""
import numpy as np
import cv2
import os
import tifffile as tif
import glob
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from pylab import mpl
from sklearn.decomposition import PCA
import random
import colorsys
from getLabelBox import get_boxes_maskes_byskimage
from PIL import Image,ImageEnhance

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False
Shadow_Analysis_Pattern_plot = False


def Normalize(array):
    '''
    Normalize the array
    '''
    data = np.copy(array)
    mx = np.nanmax(data)
    mn = np.nanmin(data)
    if mx == mn:
        # print("All-same value slice encountered")
        return np.zeros_like(data)
    t = (data-mn)/(mx-mn)
    return t


def GetLight(img):
    '''计算人眼视觉特性亮度'''
    mimg = img.copy()
    # mimg[mimg < 0.1] *= 0.4
    # mimg[mimg < 0.25] *= 0.6
    # mimg[mimg < 0.5] *= 0.6
    B = mimg[:, :, 0]
    G = mimg[:, :, 1]
    R = mimg[:, :, 2]
    # result = 0.2 * R + 0.5 * G + 0.3 * B
    result = (R+G+B)/3
    result = Normalize(result)
    plot_figure(result,'亮度')
    return result


def GetColor(img):
    '''色度空间归一化'''
    mimg = img.copy()
    misc = mimg[:, :, 0] + mimg[:, :, 1] + mimg[:, :, 2]
    misc[misc == 0] = 0.0000001
    mimg[:, :, 0] = img[:, :, 0] / misc
    mimg[:, :, 1] = img[:, :, 1] / misc
    result = np.abs(mimg - img)
    result = (result[:, :, 0] + result[:, :, 1]) / 2
    result = Normalize(result)
    plot_figure(result,'色度')
    return result


def GetVege(img):
    '''获取植被特征'''
    mimg = img.copy()
    B = mimg[:, :, 0]
    G = mimg[:, :, 1]
    R = mimg[:, :, 2]
    N = mimg[:, :, 3]
    # result = G - np.minimum(R, B)
    result = (N-R)/(R+N + 0.000000001)

    result[result < 0] = 0
    result = Normalize(result)
    plot_figure(result,'植被')

    return result


def GetWater(img):
    '''获取植被特征'''
    mimg = img.copy()
    B = mimg[:, :, 0]
    G = mimg[:, :, 1]
    R = mimg[:, :, 2]
    N = mimg[:, :, 3]
    # result = G + B - (N + R)
    result = (G-N)/(G+N+0.00000000001)
    result[result < 0] = 0
    result = Normalize(result)
    plot_figure(result, '水体')

    return result


def GetLDV(idist, ilight, ivege, iwater):
    '''总决策'''
    result = (1.05*idist -0.95*ilight) - 1*ivege - 0.6*iwater
    result[result < 0] = 0
    plot_figure(result,'final')
    return Normalize(result)


def plot_figure(data, title = ''):
    if Shadow_Analysis_Pattern_plot is not True:
        return
    print('figuer plotting--------------')
    plt.figure()
    plt.title(title, fontsize=12, fontweight='bold')
    plt.imshow(data)


def FinalTrare(img):
    '''结果后处理'''
    mimg = img.copy()
    mimg = np.uint8(Normalize(mimg) * 255)
    T, result = cv2.threshold(mimg, 0, 255, cv2.THRESH_OTSU)
    # result = cv2.medianBlur(result, 3)
    return np.array(result)


def getShadowinfo(data_array, building_shadow_buffer, data_mask, buffer_box_rc,
                  plot_shadow='', img_size = 128, delete_no_connected=False, floor=0):
    '''
    :param data_mask: 0-1 array[buffer_H, buffer_W], building(>0) & shadow(-1) mask
    :param shadow_mask: 0-1 array[buffer_H, buffer_W], shadow mask
    :return:
            shadow: 0-1 array[buffer_H, buffer_W], shadow mask
            buffer_result: 0-1 array[buffer_H, buffer_W], building & shadoow mask
            delta_ynorth：int, the delta of col_min, represents the shadow length on the north
            shadow_area: int, shadow area

    '''
    #  得到建筑物在buffer中的box和面积
    row_min, col_min, row_max, col_max = buffer_box_rc
    build_boxes, _, _, build_areas = get_boxes_maskes_byskimage(data_mask)  # 得到mask内的建筑物足迹数据，应该就一个
    assert len(build_boxes) == 1, 'mask 出错，一个连通区内出现多个建筑物范围'
    build_box = build_boxes[0]
    _, build_ymin, _, _ = build_box
    build_area = build_areas[0]

    #  得到buffer内阴影部分
    shadow = np.where(building_shadow_buffer == -1, -1, 0)
    shadow_result = shadow.copy()       # 初始化阴影mask
    shadow_area = len(shadow[shadow == -1])     # 初始化阴影面积
    buffer_ymin = build_ymin  # 初始化阴影最远长度
    build_shadow_mask_buffer = np.where(building_shadow_buffer > 0, 1, shadow)     # 合并建筑物（1）和阴影部分（-1），此时array内包含0，-1，1三种值
    buffer_maskall = np.where((shadow==-1) | (data_mask == 1), True, False)     # 得到非0区域的统一mask
    buffer_result = np.where(buffer_maskall, 1, 0)  # 初始化结果：目前是所有包含阴影区域，不排除不连通区的

    # 以下操作是为去除非联通区域以及获取最北阴影坐标的
    buffer_boxes, buffer_mask_singles, _, buffer_areas = get_boxes_maskes_byskimage(buffer_maskall,
                                                                                    mask_01_mode=True,
                                                                                    connect_mode=2)   #得到非0区域的各个mask,
                                                                                                    # mask_01_mode为False时返回的mask为布尔类型,否则0-1类型
    for i in range(len(buffer_boxes)):  # 找到与建筑物连通的那个mask
        mask = buffer_mask_singles[i]
        box = buffer_boxes[i]
        area = buffer_areas[i]
        if delete_no_connected:     # 去除不连通部分
            buffer_temp = np.where(mask == 1, build_shadow_mask_buffer, 0)   # 查看mask内对应的值
            if buffer_temp[buffer_temp == 1].any():  # 有建筑物区域：说明此处阴影与建筑物连通
                buffer_result = np.where(buffer_temp != 0, 1, 0) # 得到build+shadow的0-1mask
                shadow_result = np.where((mask == 1) & (shadow == -1), shadow, 0)
                buffer_ymin = box[1]  # 得到build+shadow的最北边的界限
                shadow_area = area - build_area  # 得到阴影面积
        else:
            buffer_ymin = min(box[1], buffer_ymin)  # 不去除非连通区域则直接用所有阴影最北像素作为结果

    # 计算阴影北向长度
    delta_ynorth = build_ymin - buffer_ymin

    shadow_result[shadow_result == -1] = 1
    shadow[shadow == -1] = 1
    if plot_shadow:  #可视化
        if shadow_area >=1 and floor == int(plot_shadow.split('_')[1]):
            color_list = ['#00000000', '#4cb4e7']
            my_cmap = LinearSegmentedColormap.from_list('mcmp', color_list)
            cm.register_cmap(cmap=my_cmap)
            plt.figure()
            plt.title(f'{floor}层阴影提取结果：{delta_ynorth}', fontsize=12, fontweight='bold')
            plt.imshow(data_array[:, :, 0:3])
            plt.imshow(shadow, cmap='mcmp')

            color_list1 = ['#00000000', '#4c00e7']
            my_cmap1 = LinearSegmentedColormap.from_list('mcmp1', color_list1)
            cm.register_cmap(cmap=my_cmap1)
            plt.imshow(data_mask, cmap='mcmp1')
            plt.figure()
            plt.imshow(data_array[:, :, 0:3])

    return shadow_result, buffer_result, delta_ynorth, shadow_area


def cut_data_save(data, savepath, positions, img_size=128):
    row_min, col_min, row_max, col_max = positions
    print(data.shape, positions)
    if len(data.shape) == 2:
        test_img = np.zeros((img_size,img_size))
        test_img[row_min:row_max, col_min:col_max] = data[:, :]

    if len(data.shape) == 3:
        test_img = np.zeros((img_size, img_size, 4))
        test_img[row_min:row_max, col_min:col_max, :] = data[:, :, :]
    tif.imsave(savepath, test_img)


def getShadows(filename, tif_file_data, building_shadow_data, label_data,
               img_size=128, shadow_area_thd=1, plot_shadow='',
               delete_no_connected=True):
    building_top_muxs = []
    building_buffer_muxs = []
    shadow_muxs = []
    building_shadow_muxs = []
    north_lengths = []
    shadow_areas = []
    floors = []

    img = tif_file_data.copy()
    label_img = label_data.copy()
    boxes, masks, labels, areas =get_boxes_maskes_byskimage(label_img, area_thd=2, connect_mode = 1)
    for i in range(len(labels)):
        box = boxes[i]
        mask = masks[i]
        floor = labels[i]
        x_min, y_min, x_max, y_max = box # west, north, east, south
        xbuf_min, ybuf_min, xbuf_max, ybuf_max = x_min - 4, y_min - int(np.ceil(floor/3) + 1), x_max + 4, y_max
        row_min, col_min, row_max, col_max = ybuf_min if ybuf_min > 0 else 0, xbuf_min if xbuf_min > 0 else 0,\
                                             ybuf_max, xbuf_max if xbuf_max < img_size else img_size# row is y, column is x
        buffer_box_rc = [row_min, col_min, row_max, col_max]
        img_buf = img[row_min:row_max, col_min:col_max, :]
        mask_buf = mask[row_min:row_max, col_min:col_max]
        building_shadow_buf = building_shadow_data[row_min:row_max, col_min:col_max]

        shadow, building_shadow, north_length, shadow_area = getShadowinfo(img_buf,
                                                                           building_shadow_buf,
                                                                           mask_buf,
                                                                           buffer_box_rc,
                                                                           plot_shadow=plot_shadow,
                                                                           delete_no_connected=delete_no_connected,
                                                                           floor=floor)

        if shadow_area < shadow_area_thd:
            continue
        # print("Shadow area==============>", shadow_area)
        building_top_mux = img_buf[mask_buf == 1]
        building_buffer_mux = img_buf[mask_buf != 1]
        shadow_mux =img_buf[shadow == 1]
        building_shadow_mux = img_buf[building_shadow == 1]

        building_top_muxs.append(building_top_mux)
        building_buffer_muxs.append(building_buffer_mux)
        shadow_muxs.append(shadow_mux)
        building_shadow_muxs.append(building_shadow_mux)
        north_lengths.append(north_length)
        shadow_areas.append(shadow_area)
        floors.append(floor)

    ShadowsInfos = {
        '建筑物层顶多光谱数据': building_top_muxs,
        '建筑物周围多光谱数据': building_buffer_muxs,
        '阴影处多光谱数据': shadow_muxs,
        '建筑物+阴影处多光谱数据': building_shadow_muxs,
        '阴影北向长度': north_lengths,
        '阴影面积': shadow_areas,
        '建筑物层数': floors,
        '文件名称': filename
    }
    return ShadowsInfos


def get_shadow_index_LDV(data):
    assert data.shape[2] == 4, 'LDV needs 4 bands (GBRN)'

    idist = GetColor(data)
    ilight = GetLight(data)
    ivege = GetVege(data)
    iwater = GetWater(data)
    final = GetLDV(idist, ilight, ivege, iwater)
    return final


def get_shadow_index_MSDI(data):
    assert data.shape[2] == 4, 'MSDI needs 4 bands (GBRN)'

    data_tmp = data.reshape(-1, data.shape[2])
    pca = PCA(n_components=1)
    pixels_transformed = pca.fit_transform(data_tmp)
    PC1 = pixels_transformed.reshape((data.shape[0], data.shape[1]))
    B = data[:, :, 0]
    G = data[:, :, 1]
    R = data[:, :, 2]
    N = data[:, :, 3]
    # print(N.shape)
    MSDI = (1 - PC1) * ((B + G) / (N + R)) * ((G-R) / (G + R))
    return MSDI


def get_shadow_index_MC3(data):
    assert data.shape[2] == 4, 'MC3 needs 4 bands (GBRN)'

    B = data[:, :, 0]
    G = data[:, :, 1]
    R = data[:, :, 2]
    N = data[:, :, 3]
    C1 = np.arctan(R / np.maximum(np.maximum(G, B), N))
    C2 = np.arctan(G / np.maximum(np.maximum(N, B), R))
    C3 = np.arctan(B / np.maximum(np.maximum(N, R), G))
    C4 = np.arctan(N / np.maximum(np.maximum(G, R), B))
    MC3 = C1*C2*C3*C4
    return MC3


def get_HSV(data, opt='RGB'):
    if opt == 'GBRN':
        RGB_data = np.concatenate((data[:, :, 2:3],  # R
                                   data[:, :, 1:2],  # G
                                   data[:, :, 0:1]   # B
                                   ), axis=2)
    elif opt == 'RGB':
        RGB_data = np.copy(data)
    else:
        print("only images with channel numer 4(GBRN) or 3(RGB) are supported.")
        return None
    # cv2.cvtColor(RGB_data, RGB_data, cv2.COLOR_BGR2Luv)
    HSV = cv2.cvtColor(np.float32(RGB_data), cv2.COLOR_BGR2HSV)
    return HSV


def get_shadow_index_NSVDI(data):
    HSV = get_HSV(data, opt='GBRN' if (data.shape[2] == 4) else 'RGB')
    # H = HSV[:, :, 0]
    S = HSV[:, :, 1]
    V = HSV[:, :, 2]
    NSVDI = (S -V) / (S + V)
    return NSVDI


def get_shadow_index_LNMPSI(data):
    assert data.shape[2] == 4, 'LNMPSI needs 4 bands (GBRN)'

    N = data[:, :, 3]
    HSV = get_HSV(data, opt='GBRN')
    H = HSV[:, :, 0]
    S = HSV[:, :, 1]
    V = HSV[:, :, 2]
    LNMPSI = np.log(N * ( (H-V)/(H+V) ) * ( (S-V)/(S+V) ) + 1)
    return LNMPSI


def shadow_extract(filename, lab_path, plot_shadow='', index='LDV'):
    img = tif.imread(filename)
    ref_path = lab_path + '\\' + filename.split('\\')[-1]
    assert os.path.isfile(ref_path), 'label文件名有误'
    ref = tif.imread(ref_path)
    ref_fp = np.where(ref > 0, 1, 0)
    # 获取阴影
    img1 = img.copy()
    RGB_img = np.concatenate((img1[:, :, 2:3],  # R
                              img1[:, :, 1:2],  # G
                              img1[:, :, 0:1]   # B
                              ), axis=2)
    plot_figure(RGB_img, '原始影像')
    img1 = img1.astype(float)
    img1 = Normalize(img1)

    if index == 'LDV':
        final = get_shadow_index_LDV(img1)  # change with other indexes
    elif index == 'MSDI':
        final = get_shadow_index_MSDI(img1)  # change with other indexes
    elif index == 'NSVDI':
        final = get_shadow_index_NSVDI(img1)  # change with other indexes
    elif index == 'MC3':
        final = get_shadow_index_MC3(img1)  # change with other indexes
    elif index == 'LNMPSI':
        final = get_shadow_index_LNMPSI(img1)  # change with other indexes
    else:
        print(f"No index{index} is available!")
        return None

    shadow = FinalTrare(final)
    if plot_shadow:
        # 可视化
        color_list = ['#00000000', '#4cb4e7']
        my_cmap = LinearSegmentedColormap.from_list('mcmp', color_list)
        cm.register_cmap(cmap=my_cmap)
        fig = plt.figure()
        plt.title('阴影提取结果', fontsize=12, fontweight='bold')
        plt.imshow(img[:, :, 0:3])
        plt.imshow(shadow, cmap='mcmp')
        color_list1 = ['#00000000', '#FF0000']
        my_cmap1 = LinearSegmentedColormap.from_list('mcmp1', color_list1)
        cm.register_cmap(cmap=my_cmap1)
        plt.imshow(ref_fp, cmap='mcmp1')

    return shadow, img, ref


def get_shadowinfos(filepath='', lab_path='', city='Xian', plot_shadow=''):
    # 获取输入图片路径
    if city == 'all':
        city = ''
    filenames = glob.glob(filepath + f'\\{city}*')
    ShadowsInfo_imgs = []


    for filename in filenames:
        assert os.path.isfile(filename), '文件名有误'

        shadow, img, ref = shadow_extract(filename, lab_path, plot_shadow=plot_shadow)
        shadow = np.where(shadow == 255, -1, shadow)
        data_buildingshadow = np.where(ref>0, ref, shadow)
        ShadowsInfo_img = getShadows(filename, img, data_buildingshadow, ref, 128, 0,
                                     plot_shadow=plot_shadow,delete_no_connected=True)
        ShadowsInfo_imgs.append(ShadowsInfo_img)
    plt.show()
    return ShadowsInfo_imgs


def single_image_plot(filename, lab_path, plot_shadow):
    ShadowsInfo_imgs = []

    assert os.path.isfile(filename), '文件名有误'

    shadow, img, ref = shadow_extract(filename, lab_path, plot_shadow=plot_shadow)
    shadow = np.where(shadow == 255, -1, shadow)
    data_buildingshadow = np.where(ref > 0, ref, shadow)
    ShadowsInfo_img = getShadows(filename, img, data_buildingshadow, ref, 128, 0,
                                 plot_shadow=plot_shadow, delete_no_connected=True)
    ShadowsInfo_imgs.append(ShadowsInfo_img)
    plt.show()

if __name__ == '__main__':

    # data = tif.imread(r'F:\ExperimentData\SEASONet\Data\image\optical\Beijing_5_12.tif')
    # get_shadow_index_MSDI(data)
    # get_shadow_index_NSVDI(data)
    # get_shadow_index_MC3(data)
    single_image_plot(filename=r'F:\ExperimentData\SEASONet\Data\image\optical\Beijing_5_12.tif',
                      lab_path=r'F:\ExperimentData\SEASONet\Data\label', plot_shadow='include_15')
