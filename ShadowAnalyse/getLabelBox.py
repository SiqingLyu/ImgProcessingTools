import numpy as np
from skimage.measure import label, regionprops
import time
import tifffile as tif
BACKGROUND = 0


def get_boxes_maskes_byskimage(data, area_thd = 1, mask_01_mode = True, connect_mode=1):
    data_arr = np.array(data)
    label_region = label(data_arr, connectivity=connect_mode, background=0)
    boxes, masks, labels, areas, nos_list = [], [], [], [], []
    for region in regionprops(label_region):
        if region.area < area_thd : continue
        # region.bbox垂直方向为x， 而目标检测中水平方向为x
        y_min, x_min, y_max, x_max = region.bbox
        # print(x_max)
        boxes.append([x_min, y_min, x_max, y_max])
        m = label_region == region.label
        # 取众数
        v_nos = np.bincount(data_arr[m]).argmax()
        nos_list.append(v_nos)
        masks.append(m)
        labels.append(v_nos)
        areas.append(region.area)
    if len(boxes) == 0:
        return None,None,None,None
    assert 0 not in labels
    masks = np.array(masks)
    if mask_01_mode:
        masks = np.where(masks, 1, 0)
    return np.array(boxes), masks, np.array(labels), np.array(areas)


def get_boxes_maskes(data):
    data_temp = np.copy(data)  # 备份数据
    data_compare = np.copy(data)  # 备份数据
    width = data_temp.shape[0]
    height = data_temp.shape[1]
    boxes = []
    maskes = []
    areas = []
    contents = []
    num = 0
    '''
        以下循环主要思路是：遍历找到第一个不为背景的像素，
        对此像素递归寻找所有相连区域，将区域内每个像素的坐标保存，
        最终得到外接矩形框坐标, 同时获取每个独立区域的mask
        另外需要注意：图像中得到的xmax、ymax等实际上是x代表行数（row），y代表列数（column），
        如果要换成坐标系则需要对应row->y, column->x
    '''
    for i in range(width):
        for j in range(height):
            if data_temp[i][j] != BACKGROUND:
                # 获取当前连通区域的BOX
                content = data_temp[i][j]
                ymin, xmin, ymax, xmax, data_temp = find_box(data_temp, i, j)
                if (xmax-xmin)+(ymax-ymin) == 0:
                    continue
                else:
                    box = [xmin, ymin, xmax+1, ymax+1]
                    # 得到mask
                    mask = data_temp ^ data_compare  # 异或运算
                    data_compare = np.copy(data_temp)
                    #得到像素个数
                    area = np.sum(mask > 0)
                    # 保存mask和box
                    boxes.append(box)
                    maskes.append(mask)
                    contents.append(content)
                    areas.append(area)
                    num += 1
                    # print("第{0}个对象，矩形框轮廓（rmin, cmin, rmax, cmax）：".format(num), box)
    boxes = np.array(boxes)
    maskes = np.array(maskes)
    maskes = np.where(maskes > 0, 1, 0)
    contents = np.array(contents)
    areas = np.array(areas)
    if len(boxes) == 0:
        return None, None, None, None
    return boxes, maskes, contents, areas


def find_box(data, x, y):
    data_temp = np.copy(data)
    xy_list = find_xy(data_temp, x, y)
    xy_arr = np.array(xy_list)
    xs = xy_arr[:, 0]
    ys = xy_arr[:, 1]
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    return xmin, ymin, xmax, ymax, data_temp


# def find_box_4(data, x, y):
#
#     data_temp = np.copy(data)
#     xmin = find_xmin(data_temp, x, y) + 1
#     data_temp = np.copy(data)
#     xmax = find_xmax(data_temp, x, y) - 1
#     data_temp = np.copy(data)
#     ymin = find_ymin(data_temp, x, y) + 1
#     data_temp = np.copy(data)
#     ymax = find_ymax(data_temp, x, y) - 1
#     return xmin,ymin,xmax,ymax,data_temp


def find_xy(data, x, y, turn_mode=4):
    '''
    this function is to get all the (x,y) in one conneted region in data field
    for example:
    input:
        data:
            [ [1 1 0 0 0 0 ]
              [0 1 0 0 1 1 ]
              [1 1 0 0 0 1 ]
              [0 0 0 1 0 0 ] ]
        x: 0
        y: 0
    output:
        xy_list = [ [0,0]
                    [0,1]
                    [1,1]
                    [2,1]
                    [2,0] ]
    in this output you can find the coordinates of the Minimum Bounding Rectangle
    :param data: in array type
    :param x: parameter x & y must be a place where data is not 0 if you want to have a result
    :param y:
    :param turn_mode: leave it as a default
            0, 1, 2, 3, 4, 5, 6, 7, 8 :
            right, left, up, down, center(start position), right-up, left-down, left-up, right-down
            \ 7   2   5 \
            \ 1   4   0 \
            \ 6   3   8 \
    :return: a list of coordinates [x,y]
    '''
    xy_list = []
    data_temp = data[x][y]
    data[x][y] = BACKGROUND
    xy_list.append([x, y])
    if y - 1 >= 0:  # right
        if data[x][y-1] != BACKGROUND and data[x][y-1] == data_temp :
            if turn_mode != 1:
                xy_list_temp = find_xy(data, x, y - 1, 0)
                xy_list += xy_list_temp
    if y + 1 < data.shape[1]:   # left
        if data[x][y+1] != BACKGROUND and data[x][y+1] == data_temp:
            if turn_mode != 0:
                xy_list_temp = find_xy(data, x, y + 1, 1)
                xy_list += xy_list_temp
    if x - 1 >= 0:  # up
        if data[x-1][y] != BACKGROUND and data[x-1][y] == data_temp:
            if turn_mode != 3:
                xy_list_temp = find_xy(data, x - 1, y, 2)
                xy_list += xy_list_temp
    if x + 1 < data.shape[0]: # down
        if data[x+1][y] != BACKGROUND and data[x+1][y] == data_temp:
            if turn_mode != 2:
                xy_list_temp = find_xy(data, x + 1, y, 3)
                xy_list += xy_list_temp
    if ((x - 1 >= 0) &
            (y + 1 < data.shape[1])):   # right-up
        if data[x-1][y+1] != BACKGROUND and data[x-1][y+1] == data_temp:
            if turn_mode != 6:
                xy_list_temp = find_xy(data, x-1, y+1, 5)
                xy_list += xy_list_temp
    if ((x + 1 < data.shape[0]) &
            (y - 1 >= 0)):  # left-down
        if data[x+1][y-1] != BACKGROUND and data[x+1][y-1] == data_temp:
            if turn_mode != 5:
                xy_list_temp = find_xy(data, x+1, y-1, 6)
                xy_list += xy_list_temp
    if ((x - 1 >= 0) &
            (y - 1 >= 0)):  # left-up
        if data[x-1][y-1] != BACKGROUND and data[x-1][y-1] == data_temp:
            if turn_mode != 8:
                xy_list_temp = find_xy(data, x-1, y-1, 7)
                xy_list += xy_list_temp
    if ((x + 1 < data.shape[0]) &
            (y + 1 < data.shape[1])):   # right-down
        if data[x+1][y+1] != BACKGROUND and data[x+1][y+1] == data_temp:
            if turn_mode != 7:
                xy_list_temp = find_xy(data, x+1, y+1, 8)
                xy_list += xy_list_temp
    return xy_list


# if __name__ == '__main__':
#     data = tif.imread(r'C:\Users\lenovo\Desktop\实验\results\all_withfootprint_result_meanbox_alltest\refXian_2_7.tif')
#     data = [[1,1,0,0,0,0],
#             [1,0,0,1,1,1,],
#             [0,2,1,1,0,0]]
#     data_arr = np.array(data)
#     boxes, maskes, contents, areas = get_boxes_maskes(data_arr)
#     print(boxes, '\n', contents, '\n', areas)
#     data_arr = np.array(data)
#     label_region = label(data_arr, connectivity=2, background=0)
#     boxes, masks, labels, areas, nos_list = [], [], [], [], []
#     for region in regionprops(label_region):
#         if region.area < 1 : continue
#         # region.bbox垂直方向为x， 而目标检测中水平方向为x
#         y_min, x_min, y_max, x_max = region.bbox
#         # print(x_max)
#         boxes.append([x_min, y_min, x_max, y_max])
#         m = label_region == region.label
#         # 取众数
#         v_nos = np.bincount(data_arr[m]).argmax()
#         nos_list.append(v_nos)
#         masks.append(m)
#         labels.append(v_nos)
#         areas.append(region.area)
#     assert len(boxes) > 0
#     assert 0 not in labels
#     print(boxes, '\n',  labels, '\n', areas)
