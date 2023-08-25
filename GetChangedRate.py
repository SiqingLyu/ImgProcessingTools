from LabelTargetProcessor import LabelTarget
import os
from tools import *
import tifffile as tif


def read_bin_diff(lab_data, building_num, diff_data, building_diff_num):
    Tar = LabelTarget(label_data=lab_data).to_target_cpu()
    if Tar is not None:
        masks = Tar['masks']
        building_num += len(masks)
        for mask in masks:
            diff_this_building = diff_data[mask==1]
            diff_rate = len(diff_this_building[diff_this_building != 0]) / len(diff_this_building)
            if len(diff_this_building) <= 2:
                continue
            if diff_rate == 1:
                building_diff_num += 1
    print("Rate at this time is :", np.around((building_diff_num/building_num), 5))
    print("Buildings and Building with Change at this time is :", building_num, building_diff_num)

    return building_num, building_diff_num


def main():
    lab_path = r'F:\ExperimentData\Zeping5YearsBuildings\RoIs\Label_bk0'
    save_path = r'F:\ExperimentData\Zeping5YearsBuildings\61Cities\allDiffs'
    lab_paths, lab_names = file_name_tif(lab_path)
    file_paths, file_names = file_name_tif(save_path)
    assert len(file_names) == len(lab_names)

    read_bin = 100

    building_num = 0
    building_diff_num = 0
    for ii in range(len(file_names)):
        file_name = file_names[ii]
        print("processing : ", file_name)
        lab_data = tif.imread(lab_paths[ii])
        diff_data = tif.imread(file_paths[ii])
        width, height = lab_data.shape
        print(width, height)
        x_bin = np.floor(width/read_bin).astype(int)
        y_bin = np.floor(height/ read_bin).astype(int)
        xs = list(range(0, x_bin*read_bin+1, read_bin))+[width]
        ys = list(range(0, y_bin*read_bin+1, read_bin))+[height]


        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                lab_data_tmp = lab_data[xs[i] : xs[i+1], ys[j]: ys[j+1]]
                diff_data_tmp = diff_data[xs[i] : xs[i+1], ys[j]: ys[j+1]]
                if np.all(lab_data_tmp == 0):
                    continue
                building_num, building_diff_num = read_bin_diff(lab_data=lab_data_tmp,
                                                                building_num=building_num,
                                                                diff_data=diff_data_tmp,
                                                                building_diff_num=building_diff_num)

    print("Rate over all time is :", np.around((building_diff_num / building_num), 7))
    print("Buildings and Building with Change over all time is :", building_num, building_diff_num)


if __name__ == '__main__':
    main()