"""
Date: 2023.08.25
Author: Lv
This code is for getting the building pixels of which the value represent the change statu.
This code is to provide the data needed in GetChangedRate
"""
import os
from tools import *
import tiffile as tif
import numpy as np

citynames = ['Beijing', 'Nanjing', 'Tianjin', 'Guangzhou', 'Chongqing', 'Haerbin', 'Hangzhou',
             'Kunming', 'Nanchang', 'Shanghai', 'Shenzhen', 'Wuhan', 'Xiamen', 'Xian', 'Zhengzhou',
             'Aomen', 'Baoding', 'Changchun', 'Changsha', 'Changzhou', 'Chengdu', 'Dalian', 'Dongguan',
             'Eerduosi', 'Foshan', 'Fuzhou', 'Guiyang', 'Haikou', 'Hefei', 'Huhehaote', 'Huizhou',
             'Jinan', 'Lanzhou', 'Lasa', 'Luoyang', 'Nanning', 'Ningbo', 'Quanzhou', 'Sanya', 'Shantou',
             'Shijiazhuang', 'Suzhou', 'Taiyuan', 'Taizhou', 'Tangshan', 'Wenzhou', 'Xianggang',
             'Xining', 'Yangzhou', 'Yinchuan', 'Zhongshan', 'Jiaxing', 'Jinhua', 'Nantong',
             'Qingdao', 'Shaoxing', 'Shenyang', 'Wuxi', 'Wuhu', 'Xuzhou', 'Zhuhai']

print(len(citynames))
city_filenames = []
for cityname in citynames:
    city_filename = cityname + '.tif'
    city_filenames.append(city_filename)
print(city_filenames)



root_path = r'F:\ExperimentData\Zeping5YearsBuildings\61Cities'
lab_path = r'F:\ExperimentData\Zeping5YearsBuildings\RoIs\Label_bk0'
save_path = r'F:\ExperimentData\Zeping5YearsBuildings\61Cities\allDiffs'
make_dir(save_path)
init_year = 2018
years = [2017, 2019, 2020, 2021]


for root, dirs, files in os.walk(os.path.join(root_path, str(init_year))):
    for file in files:
        if file in city_filenames:
            file_path = os.path.join(root, file)
            print('processing', file)
            city_thisyear = tif.imread(file_path.replace(file, file.split(".")[0] + '51N_clip.tif'))  # value = 255, bk = 0
            lab = tif.imread(os.path.join(r'F:\ExperimentData\Zeping5YearsBuildings\RoIs\Label_bk0', file))  # value = nos, bk = 0

            city_thisyear = np.where(lab != 0, city_thisyear / 255, 0)

            city_diff_all = np.zeros_like(city_thisyear)

            for year in years:
                file_other_path = file_path.replace('2018', str(year))
                city_otheryear = tif.imread(file_other_path.replace(file, file.split(".")[0] + '51N_clip.tif'))
                city_otheryear = np.where(lab != 0, city_otheryear / 255, 0)

                city_diff_tmp = np.where(city_thisyear!=city_otheryear, 1, 0)

                city_diff_all += city_diff_tmp  # 5 if all year has building, 0 if no year has building, 1-4 if some year has building
            tif.imsave(os.path.join(save_path, file), city_diff_all)


