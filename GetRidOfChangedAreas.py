import os
import tifffile as tif
import numpy as np
import sys
sys.path.append('/home/dell/lsq/LSQNetModel_generalize')

from tools import *
from LabelTargetProcessor import LabelTarget

BACKGROUND = 0

SELECTED_CITIES = ['Luoyang', 'Xian', 'Hangzhou', 'Shenzhen', 'Shenyang']

def get_var(img_size, datalist, masks):
    var_data = np.zeros([img_size, img_size])
    all_data = np.zeros([img_size, img_size])

    grey_list = []

    for img in datalist:
        img_grey = np.zeros([img_size, img_size])
        dim = img.shape[2]
        for i in range(dim):
            img_dim = img[:, :, i]
            img_grey += img_dim
        img_grey /= dim
        grey_list.append(img_grey)
        all_data += img_grey

    avg_data = all_data/len(datalist)
    for data in grey_list:
        var = (data - avg_data) ** 2
        var_data += var
    var_data /= len(datalist)

    var_buildings_this_image = []
    for mask in masks:
        var_the_building = np.mean(var_data[mask])
        var_buildings_this_image.append(var_the_building)


    return var_data, np.array(var_buildings_this_image)


def get_rid_of_changed_area(var_data, target, label, var_thd):
    '''
    inputing a group of data which locate at the same place from different seasons,
    delete the building box if it contains obvious building distruct(from optical var top5%)
    '''


    new_data = np.copy(label)
    for ii in range(len(target['masks'])):
        mask = target['masks'][ii]
        var_this_building = np.mean(var_data[mask])
        if var_this_building > var_thd:
            new_data[mask] = BACKGROUND
    return new_data


def get_all_var(root_path, filenames, lab_path):
    all_vars = []
    seasons = ['spring', 'summer', 'fall', 'winter']
    var_buildings = []
    for filename in filenames:
        # if filename.split('_')[0] not in SELECTED_CITIES:
        #     continue
        print("getting var processing ", filename)
        data_seasons = []
        lab = tif.imread(os.path.join(lab_path, filename+'.tif'))
        masks = LabelTarget(label_data=lab).to_target_cpu()['masks']

        data_seasons.append(tif.imread(os.path.join(root_path, 'optical', filename + '.tif')))
        for season in seasons:
            data_seasons.append(tif.imread(os.path.join(root_path, 'season', season, filename + '.tif')))
        var_data, var_buildings_this_image = get_var(data_seasons[0].shape[0], data_seasons, masks)
        all_vars.append(var_data)  #[arr,arr,arr....,arr]
        var_buildings.append(var_buildings_this_image)

    var_buildings = np.nan_to_num(np.hstack(var_buildings))

    var_buildings.sort()
    var_thd = [int(-len(var_buildings)*0.2)]
    print(var_buildings.shape)
    print(var_buildings)
    print(np.max(var_buildings))


    return all_vars, var_thd


def main():
    root_path = r'/home/dell/lsq/Data/Test/image'
    file_paths, filenames = file_name_tif(r'/home/dell/lsq/Data/Test/label')
    lab_path = r'/home/dell/lsq/Data/Test/label'

    save_path = r'/home/dell/lsq/Data/Test/Label_ridof'
    make_dir(save_path)

    all_vars, VAR_THD = get_all_var(root_path, filenames, lab_path)

    for ii in range(len(filenames)):
        filename = filenames[ii]
        print('Processing changed regions', filename)

        var_data = all_vars[ii]
        label = tif.imread(os.path.join(lab_path, filename+'.tif'))
        target = LabelTarget(label_data=label).to_target_cpu()
        lab_ridof = get_rid_of_changed_area(var_data, target, label, VAR_THD)
        tif.imsave(os.path.join(save_path, filename + '.tif'), lab_ridof)


if __name__ == '__main__':
    main()
