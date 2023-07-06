'''
此代码用于使用GEEMAP从GEE批量下载全国数据，截至2023/7/6还未进行测试
Author： Lv
Date: 2023/07/06
Last coded: 2023/07/06

'''

# coding:utf-8

import argparse
import csv
import math
from multiprocessing.dummy import Pool, Lock
import os
import datetime as DATETIME
import random
import geemap
# datetime.strptime(str(a),"%Y%m%d")
import warnings
from tools import *
import tiffile as tif
warnings.simplefilter('ignore', UserWarning)
import ee
import numpy as np
import urllib3


debug = False

dates = [2017,
         2018,
         2019,
         2020,
         2021]

SEASON_DATE_RANGE= {
    'spring': [90, 120],
    'summer': [180, 210],
    'autumn': [270, 300],
    'winter': [0, 30],
    'all': [0, 360]
}

dates = [2018]
NODATA = -999

def get_period(year):
    time_0 = DATETIME.date(year, 1, 1)
    time_1 = DATETIME.date(year, 12, 30)
    return [(time_0.isoformat(), time_1.isoformat())]

def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)
    return image.updateMask(mask).divide(10000)


def tif_file_check(file_path):
    data = tif.imread(file_path)
    data_pixels = data.shape[0] * data.shape[1]
    nodata_pixels = len(data[data <= NODATA])
    if nodata_pixels/data_pixels > 0.2:
        return 'too much nodata'
    elif nodata_pixels/data_pixels > 0.1:
        return 'much nodata'
    else:
        return 'acceptable'


class GEEDownloader:
    def __init__(self, loc_names: list = None, save_path: str = None, cloud_pct: int = 20):
        self.loc_names = loc_names
        self.save_path = save_path
        self.cloud_pct = cloud_pct

    def download_seasonal_s2(self, shp_file, year, period_buffer=0):
        period = get_period(year)
        for season in ['spring', 'summer', 'autumn', 'winter', 'all']:
            make_dir(f"{self.save_path}/{season}")
            try:
                s2_collection = (
                    ee.ImageCollection('COPERNICUS/S2')
                    .filterBounds(shp_file)
                    .filterDate(period[0], period[1])
                    .filter(ee.Filter.dayOfYear(SEASON_DATE_RANGE[season][0]-period_buffer if SEASON_DATE_RANGE[season][0]-period_buffer >= 0 else 0,
                                                SEASON_DATE_RANGE[season][1])+period_buffer)
                    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_pct))
                    .map(maskS2clouds)
                )
                image = s2_collection.median().select(['B2', 'B3', 'B4', 'B8']).clip(shp_file)
                save_name = f"{self.save_path}/{season}/{period[0].split('-')[0]}_s2.tif"
                geemap.download_ee_image(image, scale=10, region=shp_file, crs="EPSG:4326")
                tif_statu = tif_file_check(save_name)
                if tif_statu == 'acceptable' or period_buffer >= 30:
                    continue
                elif tif_statu == 'much nodata':
                    os.remove(save_name)
                    period_buffer_temp = period_buffer + 5
                    self.download_seasonal_s2(shp_file, year, period_buffer_temp)
                else:
                    os.remove(save_name)
                    period_buffer_temp = period_buffer + 10
                    self.download_seasonal_s2(shp_file, year, period_buffer_temp)

            except (ee.EEException, urllib3.exceptions.HTTPError) as e:
                if debug:
                    print(e)


def main(shp_path, loc_names, save_path):
    file_paths, file_names = file_name_shp(shp_path)
    for ii in range(len(file_names)):
        file_name = file_names[ii]
        file_path = file_paths[ii]
        if file_name not in loc_names:
            continue
        s2_downloader = GEEDownloader(loc_names, save_path)
        shp_file = geemap.shp_to_ee(file_path)
        s2_downloader.download_seasonal_s2(shp_file=shp_file, year=2018)


if __name__ == '__main__':
    # For users in mainland China, you need to use a VPN and configure a proxy here, because google services are not
    # accessible in China
    # For other regions, you can comment out the following
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=r"D:\try", help="Save path")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of workers for multiprocessing")
    parser.add_argument('--sample_points', type=int, default=200,
                        help="Number of sampling points around the base point ")
    parser.add_argument('--cloud_pct', type=int, default=10)
    parser.add_argument('--debug', default=False)

    args = parser.parse_args()

    ee.Authenticate()
    ee.Initialize()
    # A file containing coordinates of more than 2,
    # 000 administrative units in China to provide basic sampling information. If you want to sample in other
    # countries, you can prepare a similar document
    n = args.sample_points
    # Use Gaussian sampling to keep data concentrated in urban areas as much as possible

    def worker(idx):
        idx = idx
        for loc_id in range(n):
            if args.save_path is not None:
                location_path = os.path.join(args.save_path, f'{idx:06d}')
                os.makedirs(location_path, exist_ok=True)
                sub_location_path = os.path.join(location_path, str(loc_id))
                os.makedirs(sub_location_path, exist_ok=True)
                main(shp_path, loc_names, save_path)
        return


    indices = range(len(loc_names))

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
