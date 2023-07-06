'''
本代码中包含三个类以及一个方法，
其中第二个类 FilesMetrics 继承自第一个类，只需要输入要评价的文件所在目录，即可进行包括分层评价、分城市评价、总体评价等一系列操作
第三个类 MetricPlotter 是用于将评价结果可视化、打印出来成为图片的

需要注意的是，本代码中进行精度评价时是先对每个图（128*128为例）进行评价，然后将每个图的评价进行平均得到对于的总体评价精度，
更加稳妥的办法是将全部的对象搜集之后，统一进行评价然后再进行平均值计算，使用时需要注意两者的差别
'''

import sys
import os
import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
sys.path.append('/media/dell/shihaoze/lsq/LSQNetModel')
from skimage.measure import label, regionprops
from scipy import optimize
from matplotlib.colors import LogNorm, BoundaryNorm, NoNorm
import matplotlib.ticker as ticker

# from Data.LabelTargetProcessor import LabelTarget
ZERON = 0.00000001  # infinitesimal
from tools import make_dir, file_name_tif
from tqdm import tqdm
import tiffile as tif
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
'''
cmap
 supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r',
'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens',
'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 
'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 
'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu',
'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 
'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 
'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 
'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 
'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 
'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 
'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 
'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 
'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 
'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 
'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 
'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 
'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 
'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 
'winter', 'winter_r'

'''

CMAP='GnBu_r'
colors = ['red', 'black', 'orange', 'green', 'cyan']
alphas = [1.0, 1.0, 0.6, 0.5, 0.3]
colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
alphas = [1, 1, 1, 1, 1, 1, 1]

colors = ['red', 'black', 'orange', 'yellow', 'green', 'blue', 'purple']
alphas = [1.0, 0.7, 0.7, 1, 0.7, 0.7, 0.8]

# colors = ['red', 'black']
# alphas = [1.0, 1, 1, 1, 1, 1, 1]

def get_box_mask_value_area(label_data,
                            area_thd: int = 1,
                            mask_mode: str = 'value',
                            background: int = 0,
                            label_is_value: bool = False):
    """
    use skimage.measure to get boxes, masks and the values, the areas of them
    :param label_data:
    :param label_is_value:
    :param background: background value of the image
    :param area_thd: objects whose area is below area_thd will be discard
    :param mask_mode: whether to connect pixels by 'is not background' or values
    :return: Boxes, Masks, Labels, Areas, all in array type
    """
    assert mask_mode in ['value', '01'], 'mask_mode must in [value, 01]'
    data = np.copy(label_data)
    value_region = label(data, connectivity=2, background=background)
    boxes, masks, labels, areas, nos_list = [], [], [], [], []
    for region in regionprops(value_region):
        if region.area < area_thd: continue
        # region.bbox垂直方向为x， 而目标检测中水平方向为x
        y_min, x_min, y_max, x_max = region.bbox
        boxes.append([x_min, y_min, x_max, y_max])
        m = value_region == region.label
        # 取众数
        value = np.bincount(data[m]).argmax()
        nos_list.append(value)
        masks.append(m)
        labels.append(value if label_is_value else 1)
        areas.append(region.area)
    if len(boxes) == 0:
        return None, None, None, None, None
    assert background not in labels
    masks = np.array(masks)
    if mask_mode is '01':
        masks = np.where(masks, 1, 0)
    return np.array(boxes), masks, np.array(labels), np.array(areas), np.array(nos_list)

class RasterMetrics:
    def __init__(self):
        self.raster = None
        self.label = None
        self.RMSE = None
        self.MAE = None
        self.res_rel = None
        self.mIoU = None
        self.area_thd = 1

    def init_data(self, raster, label):
        assert raster is not None and label is not None, 'metric data is None!'
        self.raster = raster
        self.label = label
        self.RMSE = None
        self.MAE = None
        self.res_rel = None
        self.mIoU = None

    def metric_RMSE(self, data, compare):
        diff = data.flatten() - compare.flatten()
        self.RMSE = np.sqrt(np.mean(diff * diff))

    def metric_MAE(self, data, compare):
        self.MAE = np.mean(np.abs(data - compare))

    def metric_res(self, data, compare):
        self.res_rel = np.mean(np.abs(data-compare) / compare)

    def metric_mIoU(self, data, compare):
        l = len(data)
        IOU_all = 0.
        for i in range(l):
            max_ = max(data[i], compare[i])
            min_ = min(data[i], compare[i])
            IOU_all += min_ / (max_ + ZERON)
        self.mIoU = IOU_all / (l + ZERON)

    def get_metrics(self, data, compare, object_on=False):
        assert data is not None and compare is not None, 'metric data is None!'
        self.metric_RMSE(data, compare)
        self.metric_MAE(data, compare)
        if object_on:
            self.metric_res(data, compare)
            self.metric_mIoU(data, compare)

    def get_metrics_values(self, object_on=False):
        return [self.RMSE, self.MAE, self.res_rel, self.mIoU] if object_on else [self.RMSE, self.MAE]

    def get_gtpred(self):
        label_data = np.copy(self.label)
        pred_data = np.copy(self.raster).astype(np.float32)
        boxes, masks, labels, areas, nos = get_box_mask_value_area(label_data, area_thd=self.area_thd)
        if boxes is None:
            return [-1], [-1]
        pred_labels = []
        for i in range(len(masks)):
            mask = masks[i]
            m = mask > 0
            pred_mask = pred_data[m]
            pred_label = np.mean(pred_mask)
            pred_labels.append(pred_label)
        pred_labels = np.array(pred_labels)
        labels_true, labels_pred = nos, pred_labels
        return labels_true, labels_pred

    def metric_on_pixel(self):
        data = np.copy(self.raster).astype(np.float32)
        compare = np.copy(self.label).astype(np.float32)
        self.get_metrics(data, compare)

    def metric_on_object(self):
        label, predict = self.get_gtpred()
        label, predict = np.array(label), np.array(predict)
        self.get_metrics(predict, label, object_on=True)

    def resize(self, zoom_times=1):
        self.raster = zoom(input=self.raster, zoom=zoom_times, order=0)
        self.label = zoom(input=self.label, zoom=zoom_times, order=0)


class FilesMetrics(RasterMetrics):
    def __init__(self, filepath, level_list, pred_dirname='pred', label_dirname='lab', BACKGROUND=0, metric_on='pixel', dir_or_fix='dir'):
        """
        here we assume that pred files and pred files are put into
        two different dictionary(father/pred, father/label) of one father dictionary
        :param filepath: father dictionary of pred-label data
        :param level_list: level of the number of the floor
        :param BACKGROUND: the default background of the image
        """
        super(FilesMetrics, self).__init__()
        assert metric_on in ['pixel', 'object'], 'only support pixel-wise and object-wise metrics'
        self.filepath = filepath
        self.level = level_list
        self.BACKGROUND = BACKGROUND
        self.pred_dirname = pred_dirname
        self.label_dirname = label_dirname
        self.pred_filepaths = pred_dirname
        self.lab_filepaths = label_dirname
        self.filenames = None
        self.metric_on = metric_on
        self.get_all_pred_label_path()

    def get_all_pred_label_path(self):
        dir_name = self.filepath
        pred_path = os.path.join(dir_name, self.pred_dirname)
        lab_path = os.path.join(dir_name, self.label_dirname)
        pred_filepaths, pred_filenames = file_name_tif(pred_path)
        lab_filepaths, lab_filenames = file_name_tif(lab_path)
        assert pred_filenames == lab_filenames, 'pred and label is not matching'
        self.pred_filepaths = pred_filepaths
        self.lab_filepaths = lab_filepaths
        self.filenames = pred_filenames
        return pred_filepaths, lab_filepaths, pred_filenames

    def metric_bycity(self, logtxt):
        city_metrics = {}
        logtxt = logtxt[:-4] + '_city.txt'
        city_names = self.get_all_citynames()
        with open(logtxt, mode='w') as log:
            pred_path_list, lab_path_list, filename_list = self.pred_filepaths, self.lab_filepaths, self.filenames
            pbar = tqdm(city_names)
            for city in city_names:
                pbar.set_description(f"Metric by city, City: {city}")
                pbar.update()
                Metrics_allfile = []
                for ii in range(len(filename_list)):
                    file_name = filename_list[ii]
                    lab_path = lab_path_list[ii]
                    pred_path = pred_path_list[ii]
                    assert file_name in lab_path and file_name in pred_path, 'filename is not matching pathname'
                    if city not in file_name:
                        continue
                    label = tif.imread(lab_path)
                    pred = tif.imread(pred_path)
                    leveled_labels = self.get_leveled_label(label)
                    Metricss = []
                    for ii in range(len(leveled_labels)):
                        level_label = leveled_labels[ii]
                        correspond_pred = np.where(level_label != self.BACKGROUND, pred, self.BACKGROUND)
                        self.init_data(correspond_pred, level_label)
                        if self.metric_on is 'pixel':
                            self.metric_on_pixel()
                            Metric_ = self.get_metrics_values()
                        if self.metric_on is 'object':
                            self.metric_on_object()
                            Metric_ = self.get_metrics_values(object_on=True)
                        Metricss.append(Metric_)
                    Metrics_allfile.append(Metricss)
                Metrics_allfile = np.array(Metrics_allfile)
                mean_mtrics = []
                for i in range(len(level) - 1):
                    Metric_level_temp = Metrics_allfile[:, i]
                    RMSEs = Metric_level_temp[:, 0]
                    MAEs = Metric_level_temp[:, 1]
                    if self.metric_on is 'object':
                        res_rels = Metric_level_temp[:, 2]
                        mIOUs = Metric_level_temp[:, 3]
                    RMSEs = RMSEs[RMSEs > 0]
                    MAEs = MAEs[MAEs > 0]
                    if self.metric_on is 'object':
                        res_rels = res_rels[res_rels > 0]
                        mIOUs = mIOUs[mIOUs > 0]
                    RMSE_tmp = np.nanmean(RMSEs) if len(RMSEs) > 0 else 0
                    MAE_tmp = np.nanmean(MAEs) if len(MAEs) > 0 else 0
                    if self.metric_on is 'object':
                        res_rel_temp = np.nanmean(res_rels) if len(res_rels) > 0 else 0
                        mIOU_temp = np.nanmean(mIOUs) if len(mIOUs) > 0 else 0
                        mean_mtric = [RMSE_tmp, MAE_tmp, res_rel_temp, mIOU_temp]
                    else:
                        mean_mtric = [RMSE_tmp, MAE_tmp]
                    mean_mtrics.append(mean_mtric)
                log.write('City:' + city + '\nmean Metrics:' + str(mean_mtrics) + '\n')
                city_metrics[city] = mean_mtrics
        log.close()
        return city_metrics

    def get_all_citynames(self):
        filename_list = self.filenames
        city_names = []
        for ii in range(len(filename_list)):
            file_name = filename_list[ii]
            city_name = file_name.split('_')[0]
            if city_name in city_names:
                continue
            else:
                city_names.append(city_name)
        return city_names

    def get_leveled_label(self, label, level = None):
        if level is None:
            level = self.level
        leveled_labels = []
        for i in range(len(level) - 1):
            level_min = level[i]
            level_max = level[i + 1]
            leveled_label = np.where((label > level_min) & (label <= level_max), label, self.BACKGROUND)
            leveled_labels.append(leveled_label)
        return leveled_labels

    def metric_bylevel(self, logtxt):
        level = self.level
        logtxt = logtxt[:-4] + '_level.txt'
        with open(logtxt, mode='w') as log:
            pred_path_list, lab_path_list, filename_list = self.pred_filepaths, self.lab_filepaths, self.filenames
            Metrics_allfile = []
            pbar = tqdm(filename_list)
            for ii in range(len(filename_list)):
                file_name = filename_list[ii]
                pbar.set_description(f"Metric by Level, file name: {file_name}")
                pbar.update()
                pred_path = pred_path_list[ii]
                lab_path = lab_path_list[ii]
                label = tif.imread(lab_path)
                pred = tif.imread(pred_path)
                leveled_labels = self.get_leveled_label(label)
                Metricss = []
                for level_label in leveled_labels:
                    correspond_pred = np.where(level_label != self.BACKGROUND, pred, self.BACKGROUND)
                    self.init_data(correspond_pred, level_label)
                    if self.metric_on is 'pixel':
                        self.metric_on_pixel()
                        Metric_ = self.get_metrics_values()
                    if self.metric_on is 'object':
                        self.metric_on_object()
                        Metric_ = self.get_metrics_values(object_on=True)
                    Metricss.append(Metric_)
                log.write(
                    'file:' + str(file_name) + '\nlevel:' + str(level) + '\nMetrics:' + str(Metricss) + '\n\n')
                Metrics_allfile.append(Metricss)
            Metrics_allfile = np.array(Metrics_allfile)
            mean_mtrics = []
            for i in range(len(level) - 1):
                Metric_level_temp = Metrics_allfile[:, i]
                RMSEs = Metric_level_temp[:, 0]
                MAEs = Metric_level_temp[:, 1]
                if self.metric_on is 'object':
                    res_rels = Metric_level_temp[:, 2]
                    mIOUs = Metric_level_temp[:, 3]
                RMSEs = RMSEs[RMSEs > 0]
                MAEs = MAEs[MAEs > 0]
                if self.metric_on is 'object':
                    res_rels = res_rels[res_rels > 0]
                    mIOUs = mIOUs[mIOUs > 0]
                RMSE_tmp = np.nanmean(RMSEs) if len(RMSEs) > 0 else 0
                MAE_tmp = np.nanmean(MAEs) if len(MAEs) > 0 else 0
                if self.metric_on is 'object':
                    res_rel_temp = np.nanmean(res_rels) if len(res_rels) > 0 else 0
                    mIOU_temp = np.nanmean(mIOUs) if len(mIOUs) > 0 else 0
                    mean_mtric = [RMSE_tmp, MAE_tmp, res_rel_temp, mIOU_temp]
                else:
                    mean_mtric = [RMSE_tmp, MAE_tmp]
                mean_mtrics.append(mean_mtric)
            log.write('level:' + str(level) + 'mean Metrics:' + str(mean_mtrics))
        log.close()
        return mean_mtrics

    def metric_includeall(self, logtxt):
        logtxt = logtxt[:-4] + '_includeall.txt'
        with open(logtxt, mode='w') as log:
            pred_path_list, lab_path_list, filename_list = self.pred_filepaths, self.lab_filepaths, self.filenames
            Metrics_allfile = []
            pbar = tqdm(filename_list)
            for ii in range(len(filename_list)):
                file_name = filename_list[ii]
                pred_path = pred_path_list[ii]
                lab_path = lab_path_list[ii]
                pbar.set_description(f"Metric all pixels, file name: {file_name}")
                pbar.update()
                label = tif.imread(lab_path)
                pred = tif.imread(pred_path)
                self.init_data(pred, label)
                if self.metric_on is 'pixel':
                    self.metric_on_pixel()
                    Metric_ = self.get_metrics_values()
                if self.metric_on is 'object':
                    self.metric_on_object()
                    Metric_ = self.get_metrics_values(object_on=True)
                Metrics = Metric_
                log.write('file:' + str(file_name) + '\nMetrics:' + str(Metrics) + '\n\n')
                Metrics_allfile.append(Metrics)
            Metrics_allfile = np.array(Metrics_allfile)
            mean_mtrics = []
            Metric_level_temp = Metrics_allfile
            RMSEs = Metric_level_temp[:, 0]
            MAEs = Metric_level_temp[:, 1]
            # filter the -1 value
            RMSEs = RMSEs[RMSEs >= 0]
            MAEs = MAEs[MAEs >= 0]
            RMSE_tmp = np.nanmean(RMSEs)
            MAE_tmp = np.nanmean(MAEs)
            mean_mtric = [RMSE_tmp, MAE_tmp]
            mean_mtrics.append(mean_mtric)
            log.write('mean Metrics:' + str(mean_mtrics))
        log.close()
        return mean_mtrics

    def metric_bylevelall(self, logtxt):
        level = np.arange(0, 31)
        logtxt = logtxt[:-4] + '_levelall.txt'
        with open(logtxt, mode='w') as log:
            pred_path_list, lab_path_list, filename_list = self.pred_filepaths, self.lab_filepaths, self.filenames
            Metrics_allfile = []
            pbar = tqdm(filename_list)
            for ii in range(len(filename_list)):
                file_name = filename_list[ii]
                pred_path = pred_path_list[ii]
                lab_path = lab_path_list[ii]
                pbar.set_description(f"Metric by all Level, file name: {file_name}")
                pbar.update()
                label = tif.imread(lab_path)
                pred = tif.imread(pred_path)
                leveled_labels = self.get_leveled_label(label, level)
                Metricss = []
                for level_label in leveled_labels:
                    correspond_pred = np.where(level_label != self.BACKGROUND, pred, self.BACKGROUND)
                    self.init_data(correspond_pred, level_label)
                    if self.metric_on is 'pixel':
                        self.metric_on_pixel()
                        Metric_ = self.get_metrics_values()
                    if self.metric_on is 'object':
                        self.metric_on_object()
                        Metric_ = self.get_metrics_values(object_on=True)
                    Metricss.append(Metric_)
                log.write(
                    'file:' + str(file_name) + '\nlevel:' + str(level) + '\nMetrics:' + str(Metricss) + '\n\n')
                Metrics_allfile.append(Metricss)
            Metrics_allfile = np.array(Metrics_allfile)
            mean_mtrics = []
            for i in range(len(level) - 1):
                Metric_level_temp = Metrics_allfile[:, i]
                RMSEs = Metric_level_temp[:, 0]
                MAEs = Metric_level_temp[:, 1]
                if self.metric_on is 'object':
                    res_rels = Metric_level_temp[:, 2]
                    mIOUs = Metric_level_temp[:, 3]
                RMSEs = RMSEs[RMSEs > 0]
                MAEs = MAEs[MAEs > 0]
                if self.metric_on is 'object':
                    res_rels = res_rels[res_rels > 0]
                    mIOUs = mIOUs[mIOUs > 0]
                RMSE_tmp = np.nanmean(RMSEs) if len(RMSEs) > 0 else 0
                MAE_tmp = np.nanmean(MAEs) if len(MAEs) > 0 else 0
                if self.metric_on is 'object':
                    res_rel_temp = np.nanmean(res_rels) if len(res_rels) > 0 else 0
                    mIOU_temp = np.nanmean(mIOUs) if len(mIOUs) > 0 else 0
                    mean_mtric = [RMSE_tmp, MAE_tmp, res_rel_temp, mIOU_temp]
                else:
                    mean_mtric = [RMSE_tmp, MAE_tmp]
                mean_mtrics.append(mean_mtric)
            log.write('level:' + str(level) + 'mean Metrics:' + str(mean_mtrics))
        log.close()
        return mean_mtrics

    def get_all_gtpreds(self):
        pred_path_list, lab_path_list, filename_list = self.pred_filepaths, self.lab_filepaths, self.filenames
        pbar = tqdm(filename_list)
        gtpreds_all = np.array([0, 0])
        for ii in range(len(filename_list)):
            file_name = filename_list[ii]
            pred_path = pred_path_list[ii]
            lab_path = lab_path_list[ii]
            pbar.set_description(f"Getting all gtpreds, file name: {file_name}")
            pbar.update()

            label = tif.imread(lab_path)
            pred = tif.imread(pred_path)
            self.init_data(pred, label)
            labels, preds = self.get_gtpred()
            if labels is None:
                continue
            else:
                labels = np.array(labels)
                preds = np.array(preds)
                gtpreds_all = np.vstack((gtpreds_all, np.vstack((labels, preds)).transpose()))
        labels_all = gtpreds_all[:, 0]
        preds_all = gtpreds_all[:, 1]
        print("===============Label Mean: ", np.mean(labels_all), "============")
        return labels_all, preds_all


    def get_all_gtpreds_onpixel(self):
        pred_path_list, lab_path_list, filename_list = self.pred_filepaths, self.lab_filepaths, self.filenames
        pbar = tqdm(filename_list)
        labels = []
        preds = []
        for ii in range(len(filename_list)):
            file_name = filename_list[ii]
            pred_path = pred_path_list[ii]
            lab_path = lab_path_list[ii]
            pbar.set_description(f"Getting all gtpreds, file name: {file_name}")
            pbar.update()

            label = tif.imread(lab_path)
            pred = tif.imread(pred_path).astype(np.float32)
            labels.append(label)
            preds.append(pred)
            break
        labels_all = np.array(labels).flatten()
        preds_all = np.array(preds).flatten()
        return labels_all, preds_all


class MetricPlotter:
    #TODO: advance this class into a robustter one
    def __init__(self, gt=None, pred=None, losses=None, loss_list=None):
        self.gt = gt
        self.pred = pred
        self.losses = losses
        self.loss_list = loss_list

    def double_polygon_metric(self, t_uniq, figure_savepath=r'../Metrics/各个层数上评价指标分布 @ all.png', metric_on='object'):
        '''
        plot a double-line polygon figure
        :param nos_losses: loss array or list
        :return:
        '''
        if metric_on == 'object':
            loss_name = ['RMSE', 'MAE', 'res_rel', 'IoU_NoS']
            nos_df = pd.DataFrame({n: self.losses[i] for i, n in enumerate(loss_name)}, index=t_uniq)
            nos_df['nosIoU'] = nos_df['IoU_NoS']
            ax = nos_df[['MAE', 'nosIoU']].plot(figsize=(10, 10), fontsize=20, secondary_y=['nosIoU'])
            ax.set_xlabel('NoS', fontsize=20)
            ax.set_ylabel('MAE', fontsize=20)
            ax2 = ax.right_ax
            ax2.set_ylabel('nosIoU', fontsize=20)
            ax2.legend(fontsize=20, loc='upper center');
            ax.legend(fontsize=20, loc='upper right');
        else:
            loss_name = ['RMSE', 'MAE']
            nos_df = pd.DataFrame({n: self.losses[i] for i, n in enumerate(loss_name)}, index=t_uniq)
            ax = nos_df[['MAE', 'RMSE']].plot(figsize=(10, 10), fontsize=20, secondary_y=['RMSE'])
            ax.set_xlabel('RMSE', fontsize=20)
            ax.set_ylabel('MAE', fontsize=20)
            ax2 = ax.right_ax
            ax2.set_ylabel('RMSE', fontsize=20)
            ax2.legend(fontsize=20, loc='upper center');
            ax.legend(fontsize=20, loc='upper right');
        plt.savefig(figure_savepath, dpi=200);
        plt.show()
        plt.close()

    def all_scatter(self, figure_savepath=r'../Metrics/各个高度上真值预测散点图 @ all.png'):
        '''
        plot a double-line polygon figure
        :param nos_losses: loss array or list
        :return:
        '''

        x = self.gt
        y = self.pred
        y = y[(x >= 0) & (x <= 1000)]
        x = x[(x >= 0) & (x <= 1000)]

        x = x.flatten().astype(np.uint8)
        y = np.round(y.flatten()).astype(np.uint8)

        plt.figure(figsize=(5, 5), dpi=400)
        # plt.figure(figsize=(5, 5), dpi=200)
        # plt.figure(figsize=(5, 6.18), dpi=200)
        # 拟合点
        x0 = x
        y0 = y

        def f_1(x, A, B):
            return A * x + B

        # 绘制散点
        # plt.scatter(x0[:], y0[:], 3, "red")
        # 直线拟合与绘制
        A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
        x1 = np.arange(0, 90, 0.1)  # 0.01为步长
        y1 = A1 * x1 + B1
        plt.plot(x1, y1, "white")
        # plt.plot(x1, x1, "black")
        equ = 'y = ' + str(A1)[0:6] + ' * x + ' + str(B1)[0:6]
        print(equ)
        # plt.title("总分布散点图")
        # plt.xlabel('Reference floors')
        # plt.ylabel('Estimated floors')

        x = np.array(np.concatenate((x, [1000, 0]), axis=0))
        y = np.array(np.concatenate((y, [1000, 0]), axis=0))
        xmax = np.max(x)
        xmin = np.min(x)
        xbin = xmax - xmin
        ymax = np.max(y)
        ymin = np.min(y)
        ybin = ymax - ymin
        print(xmax, ymax)

        MSE = np.sum(np.power((x - y), 2)) / len(x)
        print('MSE======>', MSE, '||| VAR=====>', np.var(x), '||| mean x=====>', np.mean(x))
        R2 = 1 - MSE / np.var(x)
        RMSE = np.math.sqrt(MSE)
        # print("RMSE:", RMSE)
        print('R2: ', R2, 'RMSE: ', RMSE)
        # plt.text(2, np.max(y) - 3, f'RMSE = {np.round(RMSE, 3)}', fontsize=10, color='white')
        # plt.text(2, np.max(y) - 4, f'R2 = {np.round(R2, 3)}', fontsize=10, color='white')
        # plt.text(2, np.max(y) - 2, 'city: Nanjing', fontsize=10, color='white')
        # plt.title(text_name[0:-4])
        # h = plt.hist2d(x, y, bins=30,
        #            norm=BoundaryNorm(boundaries=[0,20,40,60,80,100,400,700,1000,3000,5000,7000,10000], ncolors=300, extend='max'))
        h = plt.hist2d(x, y, bins=(xbin, ybin), #cmap = CMAP,
                       norm=BoundaryNorm(
                           boundaries=[0,20,40,60,80,100,200,300,400,500,600,700,800,900,1000],
                           ncolors=300, extend='max')
                       )


        # plt.scatter(x, y)
        # plt.axis([0, 90, 0, 90])
        # plt.xticks([0, 15, 30, 45, 60, 75, 90], fontsize=15)
        # plt.yticks([0, 15, 30, 45, 60, 75, 90], fontsize=15)

        # cb1 = plt.colorbar(h[3], label="Samples", orientation='vertical') #, ticks=[0, 5, 10, 2000])
        # cb1.set_label('Buildings', fontsize=18)
        # tick_locator = ticker.MaxNLocator(nbins=10)  # colorbar上的刻度值个数
        # cb1.locator = tick_locator
        # cb1.set_ticks([0, 20, 40, 60, 80, 100, 300, 500, 700, 1000])
        # cb1.update_ticks()

        plt.axis([0, 30, 0, 30])
        plt.xticks([0,5,10, 15, 20,25,30], fontsize=15)
        plt.yticks([0,5,10, 15, 20,25,30], fontsize=15)
        plt.savefig(figure_savepath, dpi=200)

        plt.show()
        plt.close()

    def double_polygon_gtpred(self, t_uniq, figure_savepath=r'../Metrics/各个层数上真值预测分布 @ all.png'):
        '''
        plot a double-line polygon figure
        :param nos_losses: loss array or list
        :return:
        '''
        nos_df = pd.DataFrame({'gt': self.gt,
                               'pred': self.pred}, index=t_uniq)
        ax = nos_df[['gt', 'pred']].plot(figsize=(10, 10), fontsize=20, secondary_y=['pred'])
        ax.set_xlabel('Number of floors', fontsize=20)
        ax.set_ylabel('ground truth', fontsize=20)
        ax2 = ax.right_ax
        ax2.set_ylabel('prediction', fontsize=20)
        ax2.legend(fontsize=20, loc='upper center');
        ax.legend(fontsize=20, loc='upper right');
        plt.savefig(figure_savepath, dpi=200);
        plt.show()
        plt.close()


    def multi_polygon_superposition(self, losses_list, labels, t_uniq,
                                        figure_savepath=r'../Metrics/不同模型结果对比.png'):
            '''
            plot a double-line polygon figure
            :param nos_losses: loss array or list
            :return:
            '''
            loss_name = ['RMSE', 'MAE', 'res_rel', 'IoU_NoS']
            for loss_str in loss_name:
                y_min = 100
                y_max = 0
                plt.figure(dpi=400)
                figure_savepath = figure_savepath[:-4] + loss_str + '.png'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                for ii in range(len(losses_list)):
                    losses = losses_list[ii]
                    nos_df = pd.DataFrame({n: losses[i] for i, n in enumerate(loss_name)}, index=t_uniq)
                    nos_df[loss_str].plot( fontsize=20, color=colors[ii], label=labels[ii], alpha=alphas[ii])
                    y_max = max(y_max, np.max(nos_df[loss_str]))
                    y_min = min(y_min, np.min(nos_df[loss_str]))



                if loss_str == 'IoU_NoS':
                    plt.ylim(0,1)
                    loss_str = 'IoU$_{NoS}$'
                else:
                    plt.ylim(0, 18)
                plt.xlim(0, 31)
                plt.xlabel('Number of stories', fontsize=20)
                plt.ylabel(loss_str , fontsize=20)
                plt.legend(fontsize=16, loc='upper left')
                plt.savefig(figure_savepath, dpi=300, bbox_inches='tight');
                plt.show()
            plt.close

if __name__ == '__main__':
    paths = []
    level = [0, 1000]  # a < x <= b
    # level = [0, 5, 10, 15, 10000]
    # level = [0, 7, 14, 10000]
    # level = [0,  10,  20, 30, 10000]
    # alllevel_metrics_list = []

    # paths = []
    # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.0DynamicBuffer_88')
    # # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.1Buffer-6_70')
    # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.1Buffer-3_70')   # No Buffer
    paths.append(r'D:\Experiments\Results\V5.1Buffer0_70')  # SEASONetet
    # # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.2Nobuffer_70')
    # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.1Buffer3_70')
    # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.1Buffer6_70')
    # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.1Buffer9_70')
    paths.append(r'D:\Experiments\Results\SEASONet_SUP_decayat16_lat_40')
    # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.1Buffer21_70')
    # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.1Buffer30_70')
    # paths.append(r'D:\experiment\Experiments0\results\Maskres50_result_V5.1Buffer39_70')

    all_level_metrics_list = []
    model_names=[]
    save_path = make_dir(os.path.join('..\Metrics', 'Mask20230130'))
    print(save_path)
    # save_path = make_dir(os.path.join('../Metrics', 'Unet'))
    for path in paths:

        logtxt = os.path.join(save_path, path.split('\\')[-1] + 'object' + 'log' + str(level[0]) + str(level[1]) + '.txt')
        print('============>' + logtxt)
        metric = FilesMetrics(filepath=path, level_list=level, metric_on='object')
        # area_thd setting
        metric.area_thd = 1
        city_metrics = metric.metric_bycity(logtxt)
        leveled_metrics = metric.metric_bylevel(logtxt)

        # metric.level = level
        alllevel_metrics = metric.metric_bylevelall(logtxt)
        alllevel_metrics = np.array(alllevel_metrics).transpose()
        all_level_metrics_list.append(alllevel_metrics)
        metric.metric_on = 'pixel'
        allpixel_metrics = metric.metric_includeall(logtxt)

        gt, pred = metric.get_all_gtpreds()
        # gt, pred = metric.get_all_gtpreds_onpixel()
        # gt *= 3
        # pred *= 3
        plotter = MetricPlotter(gt=gt, pred=pred, losses=alllevel_metrics)
        # plotter = MetricPlotter(gt=gt, pred=pred)
        figure_savepath = os.path.join(save_path, path.split('/')[-1] + '各个层数上真值预测分布1_30.png')
        # plotter.double_polygon_metric(np.arange(1, 31), figure_savepath=figure_savepath, metric_on='object')
        # plotter.double_polygon_gtpred(np.arange(1, gt.shape[0]+1), figure_savepath=figure_savepath)
        figure_savepath = os.path.join(save_path, path.split('/')[-1] + '所有的真值预测散点图1_30.png')
        plotter.all_scatter(figure_savepath=figure_savepath)
        model_names.append(path.split('/')[-1])
        print('\n=====================Done====================\n',
              f'Metrics on { path.split("/")[-1]} '
               ' RMSE,MAE,res_rel,mIOU : \n'
              , np.round(leveled_metrics,3),
              '\n RMSE_ALL,MAE_ALL: ', np.round(allpixel_metrics, 3),
              '\n savepath ', save_path,
              '\n figure save path:', figure_savepath
              )
        # alllevel_metrics_list.append(alllevel_metrics)

    model_names = [ 'SEASONet', 'SEASONet_agent']

    figure_savepath = os.path.join(save_path, '不同模型结果对比.png')
    # # plotter = MetricPlotter(loss_list=alllevel_metrics_list)
    plotter.multi_polygon_superposition(all_level_metrics_list, labels=model_names, t_uniq=np.arange(1,31), figure_savepath=figure_savepath)  # only use when there are two paths
