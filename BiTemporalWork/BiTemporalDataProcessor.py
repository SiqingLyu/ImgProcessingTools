import os
import tifffile as tif
import cv2
from tools import *
import numpy as np
from fast_glcm import *
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
NODATA = -9999


class IndexCalculator:
    def __init__(self, band_arr):
        """
        :param band_arr: [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
        """
        band_arr = self.pre_process(band_arr)
        self.band_arr = band_arr
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))

        self.B2 = band_arr[0]
        self.B3 = band_arr[1]
        self.B4 = band_arr[2]

        self.B5 = band_arr[3]
        self.B6 = band_arr[4]
        self.B7 = band_arr[5]
        self.B8 = band_arr[6]

        self.B8A = band_arr[7]
        self.B11 = band_arr[8]
        self.B12 = band_arr[9]

        self.indexes = []
        self.init_indexs()

    @staticmethod
    def normalize(data):
        data = np.nan_to_num(data)
        data_statistic = data[data != NODATA]
        max_ = data_statistic.max()
        min_ = data_statistic.min()
        data = np.where(data != NODATA, (data - min_) / (max_ - min_), 0)
        assert (np.max(data) <= 1) and (np.min(data) >= 0), print(np.max(data), np.min(data))
        return data

    def pre_process(self, band_arr):
        band_arr = np.nan_to_num(band_arr)
        band_arr[band_arr == NODATA] = 0.0
        # for ii in range(len(band_arr)):
        #     band_arr[ii] = self.normalize(band_arr[ii])
        return band_arr

    def init_indexs(self):

        self.NDBI()
        self.mNDWI()
        self.NDVI()
        self.TCW()
        self.TCG()
        self.TCB()

        self.indexes.extend([self.ndbi, self.mndwi, self.ndvi, self.tcb, self.tcw, self.tcg])

    def NDBI(self):
        self.ndbi = np.nan_to_num((self.B11 - self.B8) / (self.B11 + self.B8))

    def mNDWI(self):
        self.mndwi = np.nan_to_num((self.B3 - self.B11) / (self.B3 + self.B11))

    def NDVI(self):
        self.ndvi = np.nan_to_num((self.B8 - self.B4) / (self.B4 + self.B8))

    def TCB(self):
        self.tcb = 0.3510*self.B2 + 0.3813*self.B3 + 0.3437*self.B4 + 0.7196*self.B8 + 0.2396*self.B11 + 0.1949*self.B12

    def TCG(self):
        self.tcg = -0.3599*self.B2 - 0.3533*self.B3 - 0.4737*self.B4 + 0.6633*self.B8 + 0.0087*self.B11 - 0.2856*self.B12


    def TCW(self):
        self.tcw = 0.2578*self.B2 + 0.2305*self.B3 + 0.0883*self.B4 + 0.1071*self.B8 - 0.7611*self.B11 - 0.5308*self.B12

    def dilation(self):
        dilations = []
        for band in self.band_arr:
            dilations.append(cv2.dilate(band, self.kernel))
        for index in self.indexes:
            dilations.append(cv2.dilate(index, self.kernel))
        return np.array(dilations)

    def erosion(self):
        erosions = []
        for band in self.band_arr:
            erosions.append(cv2.erode(band, self.kernel))
        for index in self.indexes:
            erosions.append(cv2.erode(index, self.kernel))
        return np.array(erosions)

    def opening(self):
        openings = []
        for band in self.band_arr:
            openings.append(cv2.morphologyEx(band, cv2.MORPH_OPEN, self.kernel))
        for index in self.indexes:
            openings.append(cv2.morphologyEx(index, cv2.MORPH_OPEN, self.kernel))
        return np.array(openings)

    def closing(self):
        closings = []
        for band in self.band_arr:
            closings.append(cv2.morphologyEx(band, cv2.MORPH_CLOSE, self.kernel))
        for index in self.indexes:
            closings.append(cv2.morphologyEx(index, cv2.MORPH_CLOSE, self.kernel))
        return np.array(closings)

    def morph_gradient(self):
        gradients = []
        for band in self.band_arr:
            gradients.append(cv2.morphologyEx(band, cv2.MORPH_GRADIENT, self.kernel))
        for index in self.indexes:
            gradients.append(cv2.morphologyEx(index, cv2.MORPH_GRADIENT, self.kernel))
        return np.array(gradients)

    def top_hat(self):
        top_hats = []
        for band in self.band_arr:
            top_hats.append(cv2.morphologyEx(band, cv2.MORPH_TOPHAT, self.kernel))
        for index in self.indexes:
            top_hats.append(cv2.morphologyEx(index, cv2.MORPH_TOPHAT, self.kernel))
        return np.array(top_hats)

    def black_hat(self):
        black_hats = []
        for band in self.band_arr:
            black_hats.append(cv2.morphologyEx(band, cv2.MORPH_BLACKHAT, self.kernel))
        for index in self.indexes:
            black_hats.append(cv2.morphologyEx(index, cv2.MORPH_BLACKHAT, self.kernel))
        return np.array(black_hats)

    def GLCMs(self):
        self.contrasts = []
        self.mean = []
        self.var = []
        self.homogeneities = []
        self.ASMs = []
        self.dissimilarities = []
        self.entropies = []
        # self.correlations = []
        for band in self.band_arr:
            band = (255 * self.normalize(band)).astype(np.uint8)
            self.mean.append(fast_glcm_mean(band, ks=13))
            self.var.append(fast_glcm_var(band, ks=13))
            self.contrasts.append(fast_glcm_contrast(band, ks=13))
            self.homogeneities.append(fast_glcm_homogeneity(band, ks=13))
            asm, energy = fast_glcm_ASM(band, ks=13)
            self.ASMs.append(asm)
            self.dissimilarities.append(fast_glcm_dissimilarity(band, ks=13))
            self.entropies.append(fast_glcm_entropy(band, ks=13))
            # glcm = greycomatrix(band, distances=[1], angles=[np.pi / 2])
            # self.correlations.append(greycoprops(glcm, 'correlation'))
            # assert greycoprops(glcm, 'correlation').shape == fast_glcm_entropy(band, ks=13).shape

        for index in self.indexes:
            index = (255 * self.normalize(index)).astype(np.uint8)
            self.mean.append(fast_glcm_mean(index, ks=13))
            self.var.append(fast_glcm_var(index, ks=13))
            self.contrasts.append(fast_glcm_contrast(index, ks=13))
            self.homogeneities.append(fast_glcm_homogeneity(index, ks=13))
            asm, energy = fast_glcm_ASM(index, ks=13)
            self.ASMs.append(asm)
            self.dissimilarities.append(fast_glcm_dissimilarity(index, ks=13))
            self.entropies.append(fast_glcm_entropy(index, ks=13))
            # glcm = greycomatrix(index, distances=[1], angles=[np.pi / 2])
            # self.correlations.append(greycoprops(glcm, 'correlation'))


        self.mean = np.array(self.mean)
        self.var = np.array(self.var)
        self.contrasts = np.array(self.contrasts)
        self.homogeneities = np.array(self.homogeneities)
        self.ASMs = np.array(self.ASMs)
        self.dissimilarities = np.array(self.dissimilarities)
        self.entropies = np.array(self.entropies)
        # self.correlations = np.array(self.correlations)


def get_building_pixels(labpath, filepath, datas):
    lab = tif.imread(labpath)
    data = tif.imread(filepath)
    data = data.transpose((2, 0, 1))
    print(data.shape)
    assert data.shape[1] == lab.shape[0] and data.shape[2] == lab.shape[1]

    for i in range(data.shape[0]):
        data_tmp = data[i]
        data_building = data_tmp[lab != 0]
        datas.append(data_building)

    return datas


def save_by_cv2(savepath, data):
    cv2.imwrite(savepath, data)
    return


def main():
    filepath = r'G:\ExperimentData\Bi-Temporal\Img\Beijing_Winter.tif'

    # filepaths, filenames = file_name_tif(r'G:\ExperimentData\Bi-Temporal\Img')
    # for ii in range(len(filepaths)):
    # filepath = filepaths[ii]
    # filename = filenames[ii]
    filepath = r'G:\ExperimentData\Bi-Temporal\Img\Beijing_Winter.tif'
    origin_img = tif.imread(filepath)
    origin_img = origin_img.transpose((2, 0, 1))
    print(origin_img.shape, origin_img[0].shape)
    Indexes = IndexCalculator(band_arr=origin_img)
    erosions = Indexes.erosion()
    dilations = Indexes.dilation()
    opening = Indexes.opening()
    closing = Indexes.closing()
    gradients = Indexes.morph_gradient()
    top_hats = Indexes.top_hat()
    black_hats = Indexes.black_hat()
    Indexes.GLCMs()
    print(black_hats.shape)
    all_result = np.concatenate((Indexes.band_arr, Indexes.ndbi[np.newaxis, :, :], Indexes.mndwi[np.newaxis, :, :],
                                 Indexes.ndvi[np.newaxis, :, :], Indexes.tcb[np.newaxis, :, :],
                                 Indexes.tcg[np.newaxis, :, :], Indexes.tcw[np.newaxis, :, :],
                                 dilations, erosions, opening, closing, gradients, top_hats, black_hats),
                                axis=0)
    print(all_result.shape)

    tif.imsave(filepath.replace('Beijing_Winter', 'Beijing_WinterGLCMFeatures'), all_result)

def main_GLCM():
    filepath = r'G:\ExperimentData\Bi-Temporal\Img\Beijing_Winter.tif'

    # filepaths, filenames = file_name_tif(r'G:\ExperimentData\Bi-Temporal\Img')
    # for ii in range(len(filepaths)):
    # filepath = filepaths[ii]
    # filename = filenames[ii]
    filepath = r'G:\ExperimentData\Bi-Temporal\Img\Beijing_Winter.tif'
    origin_img = tif.imread(filepath)
    origin_img = origin_img.transpose((2, 0, 1))
    print(origin_img.shape, origin_img[0].shape)
    Indexes = IndexCalculator(band_arr=origin_img)

    Indexes.GLCMs()

    all_result = np.concatenate((Indexes.mean, Indexes.var, Indexes.homogeneities, Indexes.contrasts,
                                 Indexes.dissimilarities, Indexes.entropies, Indexes.ASMs),
                                 axis=0)
    print(all_result.shape)

    tif.imsave(filepath.replace('Beijing_Winter', 'Beijing_WinterGLCMFeatures'), all_result)



if __name__ == '__main__':
    # main()
    labpath = r'G:\ExperimentData\Bi-Temporal\Lab\Beijing.tif'
    filepath = r'G:\ExperimentData\Bi-Temporal\Img\Beijing_WinterallFeatures_clip.tif'
    # savepath = r'G:\ExperimentData\Bi-Temporal\Img\Beijing_WinterallFeatures_clip_buildings.tif'
    datas = []

    lab = tif.imread(labpath)
    reg = lab[lab != 0]
    class_building = np.where(reg <= 6, 1, reg)
    class_building = np.where((class_building != 1) & (class_building > 6) & (class_building <= 12), 2, class_building)
    class_building = np.where((class_building != 1) & (class_building != 2), 3, class_building)
    assert np.max(class_building) == 3 and np.min(class_building) == 1
    print(len(class_building[class_building==1]), len(class_building[class_building==2]), len(class_building[class_building==3]))
    
    datas.append(reg)
    datas.append(class_building)
    for name in ['Beijing_WinterallFeatures_clip', 'Beijing_SummerallFeatures_clip', 'Beijing_WinterGLCMFeatures_clip', 'Beijing_SummerGLCMFeatures_clip']:

        filepath_tmp = filepath.replace('Beijing_WinterallFeatures_clip', name)
        print(filepath_tmp)
        # savepath_tmp = savepath.replace('Beijing_WinterallFeatures_clip_buildings', name+'_buildings')
        datas = get_building_pixels(labpath, filepath_tmp, datas)
    datas = np.array(datas)
    # print(datas.T.shape)
    # # print(datas)
    np.savetxt('G:\ExperimentData\Bi-Temporal\BiTemporalData.csv', datas.T, delimiter=',')
