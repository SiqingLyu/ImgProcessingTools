import tifffile as tif
import os
import numpy as np


def normalize(img):
    mx = np.max(img)
    mn = np.min(img)
    if mx == mn:
        return np.zeros_like(img).astype(float)
    else:
        img_ = (img - mn) / (mx - mn)
        return img_

def aggregate_images(filename, image_paths, image_size=128, save_path=''):
    img_agg = np.zeros([image_size, image_size])
    img_list = []
    for img_path in image_paths:
        img = tif.imread(img_path)
        # img = normalize(img)
        if len(img.shape) == 2:
            img_agg += img
        elif len(img.shape) == 3:
            img_grey = np.zeros([image_size, image_size])
            dim = img.shape[2]
            for i in range(dim):
                img_dim = img[:, :, i]
                img_grey += img_dim
            img_grey /= dim
            img_list.append(img_grey)
            tif.imsave(os.path.join(save_path, 'img', img_path.split('\\')[-2]+filename), img_grey)
            img_agg += img_grey
    img_agg /= len(image_paths)
    return img_agg, img_list


if __name__ == '__main__':
    paths = []
    seasons = ['spring', 'summer', 'fall', 'winter']
    root_path = r'E:\实验数据\SEASONet\Data\image'
    lab_path = r'E:\实验数据\SEASONet\Data\label'
    save_path = r'E:\实验数据\SEASONet\Aggregates'
    paths.append(os.path.join(root_path, 'optical'))
    for season in seasons:
        paths.append(os.path.join(root_path, 'season', season))

    file_name = 'Beijing_6_10.tif'
    file_name = 'Beijing_8_16.tif'
    file_name = 'Beijing_5_12.tif'

    img_paths = []
    for path in paths:
        img_paths.append(os.path.join(path, file_name))

    print(img_paths)
    aggregate_image, img_list = aggregate_images(file_name, image_paths=img_paths, save_path=save_path)

    lab_image = tif.imread(os.path.join(lab_path, file_name))
    tif.imsave(os.path.join(save_path, 'lab', file_name), lab_image)
    tif.imsave(os.path.join(save_path, 'img', file_name), aggregate_image)

    img_var = np.zeros_like(lab_image).astype(float)
    for img in img_list:
        var_tmp = (img - aggregate_image) ** 2
        img_var += var_tmp
    img_var /= len(img_list)
    tif.imsave(os.path.join(save_path, 'var', file_name), img_var)
