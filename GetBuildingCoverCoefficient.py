'''
Date: 2023/04/27
author: Lsq
Function: 这个程序用于将一张单景的（也就是泽平给到的2.5°*2.5°的影像）的建筑物足迹数据
与对应此范围的建筑物高度（事先根据对应位置裁剪好）的数据结合，根据对应的地理纬度（单景中心纬度），
以及对应需要得到的日期数据，得到考虑建筑物遮挡的建筑物系数数据。结果将会是栅格影像，背景值为0，建筑物区域的值则对应该建筑物的遮挡系数
遮挡系数：无遮挡时为1，建筑物被完全遮挡两次时为0，其他情况在此区间内执行。
本代码有两版计算方法
第一版：
例如：建筑物A本身在足迹上对应10个像素，被建筑物B遮挡了5个像素，被建筑物C遮挡了10个像素，则最终其遮挡系数应该为：1 -（10+5）/(10*2) = 0.25，最终取0.25
例如：建筑物A本身在足迹上对应10个像素，被建筑物B遮挡了5个像素，被建筑物C遮挡了10个像素，被建筑物C遮挡了5个像素， 则最终其遮挡系数应该为：1 -（10+5+5）/(10*2) = 0，最终取0
例如：建筑物A本身在足迹上对应10个像素，被建筑物B遮挡了10个像素，被建筑物C遮挡了10个像素，被建筑物C遮挡了5个像素， 则最终其遮挡系数应该为：1 -（10+10+5）/(10*2) = -0.25，最终取0
第二版：
例如：建筑物A本身在足迹上对应10个像素，其中2个未被遮挡，3个被遮挡过1次，其他5个被遮挡过两次以上，则遮挡系数为：（2+0.3*3）/10 = 0.29
这里的系数0.3可以自己改变大小，考虑的是被遮挡一次的地区依然有一定可利用太阳能。
以上，详见get_shadow_cover_rate函数。

Last coding date:2023/05/14
'''
from skimage.measure import label, regionprops
import cv2
import tifffile as tif
import numpy as np
import os
from tqdm import tqdm
import time
BACKGROUND = 0.0

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
    return path


class GetBuildingCoverCoefficient:
    def __init__(self, footprint_data: np.ndarray, buildingheight_data: np.ndarray,
                 latitude: float, imagingdate: int, resolution: float = 2.5):
        '''
        类初始化函数，主要进行参数的收集
        :param footprint_data: 建筑物足迹存储路径
        :param buildingheight_data: 建筑物高度存储路径
        :param latitude: 对应的纬度
        :param imagingdate: 对应的日期
        :param resolution: 分辨率
        '''
        self.foorprint = footprint_data
        self.buildingheight = buildingheight_data
        assert self.foorprint.shape == self.buildingheight.shape
        self.latitude = latitude
        self.date = imagingdate
        self.resolution = resolution
        self.img_size = self.foorprint.shape[0]
        self.result = None

    def assign_height_to_footprint(self, area_thd=2, value_mode='argmax', background=0, mask_mode='01'):
        '''
        将建筑物高度对应到建筑物足迹上面去，根据建筑物足迹内对应像素的众数确定此建筑物对应的高度
        :param area_thd: 这是一个阈值，如果建筑物的面积小于这个阈值则认为其过小，不予考虑
        :param value_mode: 在建筑物足迹内采用取众数还是取均值的选项
        :param background: 背景值，默认为0
        :param mask_mode: 最终得到的对应建筑物的mask的数据类型，可以选择bool类型或者0-1类型
        :return: 返回包含每个建筑物外包矩形、mask、建筑物面积、高度的数组
        '''
        footprint_data = np.copy(self.foorprint)
        height_data = np.copy(self.buildingheight)
        height_data = np.around(height_data).astype(np.uint8)
        assert value_mode in ['argmax', 'mean'], 'mask_mode must in [argmax, mean]'
        assert mask_mode in ['bool', '01'], 'mask_mode must in [bool, 01]'

        # 得到每一个建筑物足迹上对应的高度列表
        value_region = label(footprint_data, connectivity=2, background=background)
        boxes, masks, areas, height_list = [], [], [], []
        for region in regionprops(value_region):
            if region.area < area_thd: continue
            # region.bbox垂直方向为x， 而目标检测中水平方向为x
            y_min, x_min, y_max, x_max = region.bbox
            boxes.append([x_min, y_min, x_max, y_max])
            m = value_region == region.label
            if value_mode == 'argmax':
                # 取众数
                value = np.bincount(height_data[m]).argmax() if np.bincount(height_data[m]).argmax() > 0 else 1
            if value_mode == 'mean':
                # 取平均
                value = np.mean(height_data[m])
            height_list.append(value)
            masks.append(m)
            areas.append(region.area)
        if len(boxes) == 0:
            return None, None, None, None
        assert background not in height_list
        assert len(height_list) == len(masks)
        masks = np.array(masks)
        if mask_mode is '01':
            masks = np.where(masks, 1, 0)
        return np.array(boxes), masks, np.array(areas), np.array(height_list)

    def get_shadow_cover_map(self, eastwest_buffer_number=20, time_hour = 0, **kwargs):
        '''
        这是本程序的关键函数，得到建筑物的阴影覆盖次数总图像，以及建筑物的阴影覆盖率图像（即本程序总目标）
        :param eastwest_buffer_number: 阴影缓冲区向东西向扩展几个像素的选项
        :param time_hour: 需要求当天几点的数据，在6到18之间输入
        :param kwargs: 输入其他函数的参数
        '''
        boxes, masks, areas, heights = self.assign_height_to_footprint(**kwargs)
        if boxes is None:
            return None, None
        assert len(boxes) > 0
        shadow_boxes = []
        shadow_masks = []
        shadow_heights = []
        shadow_cover_map = np.zeros_like(self.foorprint).astype(np.int32)
        shadow_cover_map[shadow_cover_map == 0] = BACKGROUND
        shadow_heights_map = np.zeros_like(self.foorprint).astype(np.float32)
        shadow_heights_map[shadow_heights_map == 0.0] = BACKGROUND

        for ii in range(len(boxes)):
            height = heights[ii]
            mask = masks[ii]
            area = areas[ii]
            box = boxes[ii]
            shadow_buffer_pixels = self.get_buffer_number(height)
            x_min, y_min, x_max, y_max = box  # west, north, east, south
            building_length = x_max - x_min
            building_width = y_max - y_min

            # 扩展buffer，去掉建筑物部分，得到阴影bufferbox范围
            if shadow_buffer_pixels >= 0: #大于零说明阴影向北，否则阴影向南
                xbuf_min, ybuf_min, xbuf_max, ybuf_max = x_min - eastwest_buffer_number, y_min - shadow_buffer_pixels, x_max + eastwest_buffer_number, y_max
                # print("阴影朝向：北方")
            else:
                xbuf_min, ybuf_min, xbuf_max, ybuf_max = x_min - eastwest_buffer_number, y_min, x_max + eastwest_buffer_number, y_max - shadow_buffer_pixels
                # print("阴影朝向：南方")

            # 图像条件约束
            assert ybuf_min <= ybuf_max
            y_min, x_min, y_max, x_max = ybuf_min if ybuf_min > 0 else 0,\
                                         xbuf_min if xbuf_min > 0 else 0, \
                                         ybuf_max if ybuf_max < self.img_size else self.img_size,\
                                         xbuf_max if xbuf_max < self.img_size else self.img_size  # row is y, column is x

            # 得到shadowbufferbox
            shadow_box = [x_min, y_min, x_max, y_max]
            shadow_boxes.append(shadow_box)

            # 得到shadow的mask
            shadow_polys = self.get_shadow_poly_points(box, shadow_box, building_length, building_width, time_hour, shadow_buffer_pixels>=0)
            shadow_mask = self.get_shadow_mask(shadow_polys, mask)  #得到此建筑物对应的shadow mask
            shadow_height = self.get_shadow_height_change(shadow_mask, shadow_polys, height, box, if_north=shadow_buffer_pixels>=0)  #得到此建筑物对应的shadow mask以及它在不同地方的高度信息

            # 向整体整合
            shadow_cover_map += shadow_mask  # only when BACKGROUND=0
            shadow_heights_map = np.where(shadow_height > shadow_heights_map, shadow_height, shadow_heights_map)

        shadow_cover_rate = self.get_shadow_cover_rate(shadow_cover_map,shadow_heights_map, masks, areas, heights)
        self.result = shadow_cover_rate
        return shadow_cover_map, shadow_cover_rate

    def get_shadow_height_change(self, shadow_mask, shadow_polys, building_height, box, if_north=True):
        shadow_points = np.array([shadow_polys[2],shadow_polys[3]])
        shadow_height = np.zeros_like(self.foorprint).astype(np.float32)
        shadow_height[shadow_height == 0.0] = BACKGROUND
        if if_north:
            shadow_northmost = np.min(shadow_points[:, 1])
            building_northmost = box[1]
            north_distance = int(building_northmost-shadow_northmost)
            assert north_distance >= 0
            shadow_height[shadow_mask != BACKGROUND] = building_height/2

            if north_distance <= 1:
                return shadow_height
            else:
                height_bin = building_height/(north_distance*2)
                for i in range(north_distance):
                    shadow_y = building_northmost - i  # north -1,  south +1
                    height_temp = building_height - (i*2 + 1)*height_bin
                    shadow_temp = shadow_height
                    shadow_height[shadow_y, :] = height_temp
                    shadow_height[shadow_temp == BACKGROUND] = BACKGROUND
        else:
            shadow_southmost = np.max(shadow_points[:, 1])
            building_southmost = box[3]
            south_distance = shadow_southmost-building_southmost
            assert south_distance >= 0
            shadow_height[shadow_mask != BACKGROUND] = building_height / 2

            if south_distance <= 1:
                return shadow_height
            else:
                height_bin = building_height / (south_distance * 2)
                for i in range(south_distance):
                    shadow_y = south_distance + i  # north -1,  south +1
                    height_temp = building_height - (i * 2 + 1) * height_bin
                    shadow_temp = shadow_height
                    shadow_height[shadow_y, :] = height_temp
                    shadow_height[shadow_temp == BACKGROUND] = BACKGROUND
        return shadow_height

    def get_shadow_poly_points(self, box, shadow_box, building_length, building_width, time_hour, if_shadow_in_north, index=0):
        '''
        得到每个建筑物对应的阴影的多边形坐标
        :param box: 此建筑物的外包矩形框坐标
        :param shadow_box: 考虑建筑物阴影的新的外包矩形框
        :param building_length: 建筑物的长度
        :param building_width: 建筑物的宽度
        :param time_hour: 需要求几点的数据
        :param if_shadow_in_north: 此时阴影是否朝北
        :return:
        '''
        time_hour_compute = time_hour - 12
        shadow_box_left = shadow_box[0]
        shadow_box_top = shadow_box[1]
        shadow_box_right = shadow_box[2]
        shadow_box_bottom = shadow_box[3]
        building_left = box[0]
        building_top = box[1]
        building_right = box[2]
        building_bottom = box[3]
        adjust_point = None

        if time_hour_compute == 0:
            # 第一种情况，阴影垂直朝北
            if if_shadow_in_north:
                shadow_left_point = (building_left, shadow_box_top)
                building_left_point = (building_left, building_bottom)
                shadow_right_point = (building_right, shadow_box_top)
                building_right_point = (building_right, building_bottom)
            # 第二种情况，阴影垂直朝南
            else:
                shadow_left_point = (building_left, shadow_box_bottom)
                building_left_point = (building_left, building_top)
                shadow_right_point = (building_right, shadow_box_bottom)
                building_right_point = (building_right, building_top)
        elif time_hour_compute < 0:
            azimuth_angle = - time_hour_compute * 15
            x_margin = building_left - shadow_box_left
            # 第三种情况，阴影朝向西北
            if if_shadow_in_north:
                building_left_point = (building_left, building_bottom)
                building_right_point = (building_right, building_top)
                y_margin = building_top - shadow_box_top
                supposed_shadow_left_ybot = building_bottom - x_margin * np.tan(azimuth_angle)  # supposed shadow left top:(shadow_box_left, supposed_shadow_left_y)
                supposed_shadow_left_ytop = building_top - (x_margin + building_length) * np.tan(azimuth_angle)  # supposed shadow left top:(shadow_box_left, supposed_shadow_left_y)
                if supposed_shadow_left_ybot > shadow_box_top:
                    shadow_left_point = (shadow_box_left, supposed_shadow_left_ybot)
                else:
                    shadow_left_point = (building_left - (y_margin+building_width) / np.tan(azimuth_angle), shadow_box_top)

                if supposed_shadow_left_ytop > shadow_box_top:
                    shadow_right_point = (shadow_box_left, supposed_shadow_left_ytop)
                else:
                    shadow_right_point = (building_right - y_margin / np.tan(azimuth_angle), shadow_box_top)
                    if supposed_shadow_left_ybot > shadow_box_top:
                        adjust_point = (shadow_box_left, shadow_box_top)

            # 第四种情况，阴影朝向西南
            else:
                building_left_point = (building_left, building_top)
                building_right_point = (building_right, building_bottom)
                y_margin = shadow_box_bottom - building_bottom
                supposed_shadow_left_ytop = building_top + (x_margin + building_length) * np.tan(azimuth_angle)  # supposed shadow left top:(shadow_box_left, supposed_shadow_left_y)
                supposed_shadow_left_ybot = building_bottom + x_margin * np.tan(azimuth_angle)  # supposed shadow left top:(shadow_box_left, supposed_shadow_left_y)
                if supposed_shadow_left_ytop < shadow_box_bottom:
                    shadow_left_point = (shadow_box_left, supposed_shadow_left_ytop)
                else:
                    shadow_left_point = (building_left - (y_margin + building_width) / np.tan(azimuth_angle), shadow_box_bottom)

                if supposed_shadow_left_ybot < shadow_box_bottom:
                    shadow_right_point = (shadow_box_left, supposed_shadow_left_ybot)
                else:
                    shadow_right_point = (building_right - y_margin / np.tan(azimuth_angle), shadow_box_bottom)
                    if supposed_shadow_left_ytop < shadow_box_bottom:
                        adjust_point = (shadow_box_left, shadow_box_bottom)
        elif time_hour_compute > 0:
            azimuth_angle = 90 - time_hour_compute * 15
            x_margin = shadow_box_right - building_right
            # 第五种情况，阴影朝向东北
            if if_shadow_in_north:
                building_left_point = (building_left, building_top)
                building_right_point = (building_right, building_bottom)
                y_margin = building_top - shadow_box_top
                supposed_shadow_right_ytop = building_top - (x_margin + building_length) * np.tan(azimuth_angle)  # supposed shadow left top:(shadow_box_left, supposed_shadow_left_y)
                supposed_shadow_right_ybot = building_bottom - x_margin * np.tan(azimuth_angle)  # supposed shadow left top:(shadow_box_left, supposed_shadow_left_y)
                if supposed_shadow_right_ytop > shadow_box_top:
                    shadow_left_point = (shadow_box_right, supposed_shadow_right_ytop)
                else:
                    shadow_left_point = (building_left + y_margin / np.tan(azimuth_angle), shadow_box_top)

                if supposed_shadow_right_ybot > shadow_box_top:
                    shadow_right_point = (shadow_box_right, supposed_shadow_right_ybot)
                else:
                    shadow_right_point = (building_right + (y_margin+building_width) / np.tan(azimuth_angle), shadow_box_top)
                    if supposed_shadow_right_ytop > shadow_box_top:
                        adjust_point = (shadow_box_right, shadow_box_top)
            # 第六种情况，阴影朝向东南
            else:
                building_left_point = (building_left, building_bottom)
                building_right_point = (building_right, building_top)
                y_margin = shadow_box_bottom - building_bottom
                supposed_shadow_left_ytop = building_top + x_margin * np.tan(azimuth_angle)  # supposed shadow left top:(shadow_box_left, supposed_shadow_left_y)
                supposed_shadow_left_ybot = building_bottom + (x_margin + building_length) * np.tan(azimuth_angle)  # supposed shadow left top:(shadow_box_left, supposed_shadow_left_y)
                if supposed_shadow_left_ybot < shadow_box_bottom:
                    shadow_left_point = (shadow_box_right, supposed_shadow_left_ybot)
                else:
                    shadow_left_point = (building_left + y_margin / np.tan(azimuth_angle), shadow_box_bottom)

                if supposed_shadow_left_ytop < shadow_box_bottom:
                    shadow_right_point = (shadow_box_right, supposed_shadow_left_ytop)
                else:
                    shadow_right_point = (building_right + (y_margin+building_width) / np.tan(azimuth_angle), shadow_box_bottom)
                    if supposed_shadow_left_ybot < shadow_box_bottom:
                        adjust_point = (shadow_box_right, shadow_box_bottom)
        shadow_left_point = [np.around(shadow_left_point[i]) for i in range(2)]
        shadow_right_point = [np.around(shadow_right_point[i]) for i in range(2)]
        if adjust_point is not None:
            return [list(building_left_point), list(building_right_point), list(shadow_right_point), list(shadow_left_point), list(adjust_point)]
        else:
            return [list(building_left_point), list(building_right_point),  list(shadow_right_point), list(shadow_left_point)]



    def get_shadow_mask(self, shadow_polys, mask):
        '''
        根据多边形坐标，得到阴影填充的mask，主要使用的时cv2自带的fillPoly
        :param shadow_polys: 阴影的多边形坐标
        :param mask: 建筑物的mask
        '''
        shadow_mask = np.zeros_like(mask)
        shadow_region = np.array(shadow_polys).astype(np.int32)
        cv2.fillConvexPoly(shadow_mask, shadow_region, [1])
        shadow_mask[mask == 1] = 0
        return shadow_mask

    def get_shadow_cover_rate(self, shadow_cover_map, shadow_heights_map, masks, areas, heights):
        '''
        根据阴影的覆盖次数、建筑物本身的mask、以及其面积、阴影高度，得到建筑物被覆盖率
        '''
        assert len(areas) == len(masks)
        rate_map = np.zeros_like(self.foorprint).astype(np.float32)
        for ii in range(len(areas)):
            area = areas[ii]
            assert area > 0
            mask = masks[ii]
            height = heights[ii]
            ## 以下是第一版计算方法
            # cover_num = np.sum(shadow_cover_map[mask == 1]) if np.sum(shadow_cover_map[mask == 1]) <= (2*area) else (2*area)
            # rate = 1 - (cover_num / (2*area))
            # assert (rate >= 0) and (rate <= 1)
            # rate_map[mask == 1] = rate

            #以下是第二版计算方法，以建筑物不被阴影覆盖的部分为主要参考条件，结合阴影考虑
            shadow_height_on_building = shadow_heights_map[mask == 1]  # 得到建筑物足迹内的阴影高度分布
            shadow_cover_on_building = shadow_cover_map[mask == 1]  # 得到建筑物足迹内的阴影覆盖次数分布
            shadow_cover_on_building[shadow_height_on_building < height] = 0  # 建筑物若比阴影高，则这个地方视为不被覆盖
            cover_once = len(shadow_cover_on_building[shadow_cover_on_building == 1])
            no_cover = len(shadow_cover_on_building[shadow_cover_on_building == 0])
            no_cover_area = cover_once*0.3 + no_cover
            rate = no_cover_area / area  # 最后阴影系数就是建筑物不被覆盖区域占总区域的比例
            rate_map[mask == 1] = rate

        return rate_map

    def get_buffer_number(self, height=20):
        '''
        根据建筑高度、日期、纬度，得到对应的建筑物阴影在正午时所需要在图像上拓展的像素个数
        :param height: 建筑高度
        '''
        height_fatctor = height / self.resolution
        result = []  # [days to summer solstice, latitude]

        result.append(self.date-175) #将365天的序号改为距离夏至日的距离
        result.append(self.latitude)
        result = np.array(result).astype('float')
        sun_angle = (1 - result[0] / 90) * 23.5 #根据夏至日的距离得到太阳直射位置纬度
        if (result[1] - sun_angle) > 0:
            sun_elevation_noon = 90 - (result[1] - sun_angle)
            buffer_number = height_fatctor / (np.tan(sun_elevation_noon))
        else:
            sun_elevation_noon =90 - (sun_angle - result[1])
            buffer_number = - height_fatctor / (np.tan(sun_elevation_noon))
        return np.around(buffer_number)

    def get_result(self):
        if self.result is not None:
            return self.result
        else:
            return np.zeros_like(self.foorprint).astype(np.float32)


def make_shape_same(data1: np.ndarray, data2: np.ndarray):
    shape1 = data1.shape
    shape2 = data2.shape
    shape_ = [min(shape1[0], shape2[0]), min(shape1[1], shape2[1])]
    new_data1 = data1[:shape_[0], : shape_[1]]
    new_data2 = data2[:shape_[0], : shape_[1]]
    return new_data1, new_data2


def foorprint_process(data):
    '''
    对足迹数据进行腐蚀与膨胀处理
    :return: 返回处理后的足迹数据
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # ksize=5,5
    # erode
    img_temp = cv2.erode(data, kernel, iterations=2)
    # dilate
    return cv2.dilate(img_temp, kernel, iterations=2)


def main(footprint_data, buildingheight_data):
    CoverGetter = GetBuildingCoverCoefficient(footprint_data, buildingheight_data,latitude=40, imagingdate=360)
    CoverGetter.get_shadow_cover_map(eastwest_buffer_number=20, time_hour=9)
    return CoverGetter.get_result()


if __name__ == '__main__':
    footprint_path = r'D:\Dataset\Beijing_xiaotong\footprint.tif'
    height_path = r'D:\Dataset\Beijing_xiaotong\buildingheight.tif'
    save_path = r'D:\Dataset\Beijing_xiaotong\result_2.tif'
    # data preprocessing
    print('------------数据预处理中--------------\n')
    footprint_data = cv2.imread(footprint_path, -1)
    foorprint_process(footprint_data)
    buildingheight_data = cv2.imread(height_path, -1)
    footprint_data, buildingheight_data = make_shape_same(footprint_data, buildingheight_data)
    print('-------------数据分块中---------------\n')
    # patch processing
    PATCH_SIZE = 200
    IMG_SHAPE = footprint_data.shape
    row = int(np.floor(IMG_SHAPE[0] / PATCH_SIZE))
    row_remain = IMG_SHAPE[0] - row * PATCH_SIZE
    column = int(np.floor(IMG_SHAPE[1] / PATCH_SIZE))
    column_remain = IMG_SHAPE[1] - column * PATCH_SIZE
    final_result = np.zeros_like(footprint_data).astype(np.float32)
    print('-------------主程序开始---------------\n')
    time.sleep(1)
    with tqdm(total=(row*column+row_remain+column_remain+1)) as pbar:
        pbar.set_description('Processing patches:')
        for i in range(row):
            for j in range(column):
                footprint_patch = footprint_data[i*PATCH_SIZE: (i+1)*PATCH_SIZE, j*PATCH_SIZE: (j+1)*PATCH_SIZE]
                buildingheight_patch = buildingheight_data[i*PATCH_SIZE: (i+1)*PATCH_SIZE, j*PATCH_SIZE: (j+1)*PATCH_SIZE]
                patch_result = main(footprint_data=footprint_patch, buildingheight_data=buildingheight_patch)
                final_result[i*PATCH_SIZE: (i+1)*PATCH_SIZE, j*PATCH_SIZE: (j+1)*PATCH_SIZE] = patch_result
                pbar.update()
            if column_remain>0:
                footprint_patch = footprint_data[i * PATCH_SIZE: (i + 1) * PATCH_SIZE, column * PATCH_SIZE: IMG_SHAPE[1]]
                buildingheight_patch = buildingheight_data[i * PATCH_SIZE: (i + 1) * PATCH_SIZE, column * PATCH_SIZE: IMG_SHAPE[1]]
                patch_result = main(footprint_data=footprint_patch, buildingheight_data=buildingheight_patch)
                final_result[i * PATCH_SIZE: (i + 1) * PATCH_SIZE, column * PATCH_SIZE: IMG_SHAPE[1]] = patch_result
                pbar.update()

        if row_remain > 0:
            for j in range(column):
                footprint_patch = footprint_data[row * PATCH_SIZE: IMG_SHAPE[1], j*PATCH_SIZE: (j+1)*PATCH_SIZE]
                buildingheight_patch = buildingheight_data[row * PATCH_SIZE: IMG_SHAPE[1], j*PATCH_SIZE: (j+1)*PATCH_SIZE]
                patch_result = main(footprint_data=footprint_patch, buildingheight_data=buildingheight_patch)
                final_result[row * PATCH_SIZE: IMG_SHAPE[1], j*PATCH_SIZE: (j+1)*PATCH_SIZE] = patch_result
                pbar.update()

        if row_remain > 0 and column_remain > 0:
            footprint_patch = footprint_data[row * PATCH_SIZE: IMG_SHAPE[1], column * PATCH_SIZE: IMG_SHAPE[1]]
            buildingheight_patch = buildingheight_data[row * PATCH_SIZE: IMG_SHAPE[1], column * PATCH_SIZE: IMG_SHAPE[1]]
            patch_result = main(footprint_data=footprint_patch, buildingheight_data=buildingheight_patch)
            final_result[row * PATCH_SIZE: IMG_SHAPE[1], column * PATCH_SIZE: IMG_SHAPE[1]] = patch_result
            pbar.update()

    print('-------------结果保存中---------------\n')
    tif.imsave(save_path, final_result.astype(np.float32))

