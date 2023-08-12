#!/usr/bin/python
# -*- coding: UTF-8 -*-
from getShadows_v2 import get_shadowinfos
import numpy as np
import matplotlib.pyplot as plt
from tools import make_dir
from tqdm import tqdm


def get_length_byfloor(lengths, floors, level):
    assert len(lengths) == len(floors), 'Error: 数据长度不一致！'
    leveled_avglengths = []
    for i in range(len(level) -1 ):
        lengths_thislevel = []
        from_ = level[i]
        to_ = level[i+1]
        for n in range(len(floors)):
            floor = floors[n]
            if from_ <= floor < to_:
                lengths_thislevel.append(lengths[n])
        if len(lengths_thislevel) > 0:
            avglen_thislevel = np.mean(lengths_thislevel)
        else:
            avglen_thislevel = 0
        leveled_avglengths.append(avglen_thislevel)
    return leveled_avglengths


def get_lengthareas_season_ofthecity(seasons_paths, lab_path, city, level):
    '''
    :param seasons_paths: [spring, summer, fall, winter], str, dir_path
    :param city: city to analysis, analysis all when  = ''
    :param level: floor levels
    :return: seasons_lengths: array[4, len(level)], length-season data
    '''
    seasons_lengths = []
    seasons_areas = []
    for path in seasons_paths:
        ShadowsInfos = get_shadowinfos(path, lab_path, city, False)
        '''
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
        '''
        leveled_lengths_whole = []
        leveled_areas_whole = []
        for ShadowsInfo in ShadowsInfos:
            # roof_muxs = ShadowsInfo['建筑物层顶多光谱数据']
            # surround_muxs = ShadowsInfo['建筑物周围多光谱数据']
            shadow_lengths = ShadowsInfo['阴影北向长度']
            shadow_areas = ShadowsInfo['阴影面积']
            floors = ShadowsInfo['建筑物层数']
            # filename = ShadowsInfo['文件名称']

            leveled_lengths_patch = get_length_byfloor(shadow_lengths, floors, level)
            leveled_areas_patch = get_length_byfloor(shadow_areas, floors, level)
            leveled_lengths_whole.append(leveled_lengths_patch)
            leveled_areas_whole.append(leveled_areas_patch)
        leveled_lengths_whole = np.array(leveled_lengths_whole)
        leveled_areas_whole = np.array(leveled_areas_whole)
        # print(leveled_lengths_whole)
        avg_lengths = leveled_lengths_whole.mean(axis=0)
        avg_areas = leveled_areas_whole.mean(axis=0)
        seasons_lengths.append(avg_lengths)
        seasons_areas.append(avg_areas)
    seasons_lengths = np.array(seasons_lengths)
    seasons_areas = np.array(seasons_areas)
    return seasons_lengths, seasons_areas


def plot_length_season(seasons_lengths, save=None, city=''):
    names = ['Spring', 'Summer', 'Fall', 'Winter']
    x = range(len(names))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
    plt.plot(x, seasons_lengths[0], color='green', marker='o', linestyle='-', label='floor: 1-7')
    plt.plot(x, seasons_lengths[1], color='blue', marker='D', linestyle='-', label='floor: 7-14')
    plt.plot(x, seasons_lengths[2], color='orangered', marker='*', linestyle='-', label='floor: 15+')
    # plt.plot(x, seasons_lengths[3], color='cyan', marker='v', linestyle='-', label='floor: 16-20')

    plt.legend()  # 显示图例
    plt.title(city)  # 显示标题
    plt.xticks(x, names, rotation=45)
    plt.xlabel("Season")  # X轴标签
    plt.ylabel("Shadow Length")  # Y轴标签
    if save is not None:
        plt.savefig(save + f'/shadowlength-season.png', dpi=300)
    plt.show()
    plt.close()


def plot_length_area(seasons_areas, save=None, city=''):
    names = ['Spring', 'Summer', 'Fall', 'Winter']
    x = range(len(names))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
    plt.plot(x, seasons_areas[0], color='green', marker='o', linestyle='-', label='floor: 1-7')
    plt.plot(x, seasons_areas[1], color='blue', marker='D', linestyle='-', label='floor: 7-14')
    plt.plot(x, seasons_areas[2], color='orangered', marker='*', linestyle='-', label='floor: 15+')
    # plt.plot(x, seasons_areas[3], color='cyan', marker='v', linestyle='-', label='floor: 16-20')

    plt.legend()  # 显示图例
    plt.title(city)  # 显示标题
    plt.xticks(x, names, rotation=45)
    plt.xlabel("Season")  # X轴标签
    plt.ylabel("Shadow Area")  # Y轴标签

    if save is not None:
        plt.savefig(save + f'/shadowarea-season.png', dpi=300)
    plt.show()
    plt.close()


def main():
    lab_path = r'D:\PythonProjects\DataProcess\Data\label_bkas0'
    seasons_paths = []
    seasons_paths.append(r'D:\PythonProjects\DataProcess\Data\image\season\spring')    # spring
    seasons_paths.append(r'D:\PythonProjects\DataProcess\Data\image\season\summer')    # summer
    seasons_paths.append(r'D:\PythonProjects\DataProcess\Data\image\season\autumn')    # autumn
    seasons_paths.append(r'D:\PythonProjects\DataProcess\Data\image\season\winter')    # winter
    # city = 'Kunming'
    citynames = ['Beijing', 'Nanjing', 'Tianjin', 'Guangzhou', 'Chongqing', 'Haerbin', 'Hangzhou',
                 'Kunming', 'Nanchang', 'Shanghai', 'Shenzhen', 'Wuhan', 'Xiamen', 'Xian', 'Zhengzhou',
                 'Aomen', 'Baoding', 'Changchun', 'Changsha', 'Changzhou', 'Chengdu', 'Dalian', 'Dongguan',
                 'Eerduosi', 'Foshan', 'Fuzhou', 'Guiyang', 'Haikou', 'Hefei', 'Huhehaote', 'Huizhou',
                 'Jinan', 'Lanzhou', 'Lasa', 'Luoyang', 'Nanning', 'Ningbo', 'Quanzhou', 'Sanya', 'Shantou',
                 'Shijiazhuang', 'Suzhou', 'Taiyuan', 'Taizhou', 'Tangshan', 'Wenzhou', 'Xianggang',
                 'Xining', 'Yangzhou', 'Yinchuan', 'Zhongshan', 'all']  # 'Shenyang',
    # citynames = ['Nanjing']
    pbar = tqdm(citynames)
    for city in citynames:
        pbar.set_description(f"Analysis by city, City: {city}")
        pbar.update()
        # level = np.arange(1,31)
        # level = [1, 6, 11, 16, 21, 10000]
        level = [1, 7, 14, 40, 10000]
        seasons_lengths, seasons_areas = get_lengthareas_season_ofthecity(seasons_paths, lab_path, city, level)
        seasons_lengths = seasons_lengths[:, 0:4]
        seasons_areas = seasons_areas[:, 0:4]
        plot_length_season(seasons_lengths, save=make_dir(f'./Analysis/{city}'), city=city)
        plot_length_area(seasons_areas, save=make_dir(f'./Analysis/{city}'), city=city)



if __name__ == '__main__':
    main()