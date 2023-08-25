"""
Date: 2023.08.25
Author: Lv
This code aims to find the buildings which were predicted to be the same height in the existing product,
but different stories in our SEASONet results.
Note that the height product will be around into int to make sure the comparison is executable.
Note that the filename in height dir and in story dir should be coupled, i.e., the same name but different root.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from LabelTargetProcessor import LabelTarget
from tools import *
import matplotlib.pyplot as plt


def get_height_dict_from_product(aim_height: int, path: str, lab_path: str):
    """
    This function can find all the buildings with the aim_height
    Return a list that contains dicts:
    [ {XXX: [[building mask 1], [building mask 2], ......]},
      {YYY: [[building mask 1], [building mask 2], ......]},
      {ZZZ: [[building mask 1], [building mask 2], ......]},
    ]
    each building mask has the size of the image, e.g., [128, 128] and the pixel values is 0(background) or 1(building)

    """
    filepaths, filenames = file_name_tif(path)
    all_results = []
    for ii in range(len(filenames)):
        filename = filenames[ii]
        filepath = filepaths[ii]
        file_data = tif.imread(filepath)
        lab_data = tif.imread(os.path.join(lab_path, filename+'.tif'))
        Tar = LabelTarget(label_data=lab_data).to_target_cpu()
        masks = Tar['masks']
        aim_masks = []
        for jj in range(len(masks)):
            mask = masks[jj]
            height = np.around(np.mean(file_data[mask == 1]))
            if height == aim_height:
                aim_masks.append(masks[jj])
        if len(aim_masks) > 0:
            result = {filename: aim_masks}
            all_results.append(result)
    return all_results


def get_story_list_by_height_results(height_results: list, path: str):
    stories = []
    for height_result in height_results:
        filename = list(height_result.keys())[0]
        masks = height_result[filename]
        filepath = os.path.join(path, filename+'.tif')
        story_data = tif.imread(filepath)
        for ii in range(len(masks)):
            mask = masks[ii]
            # print(story_data[mask == 1])
            story = np.around(np.mean(story_data[mask == 1])).astype(int)
            stories.append(story)
    return stories


def plot_story_diffs(stories, heights):
    plt.figure(dpi=400)
    plt.boxplot(stories, labels=heights, meanline=True, showmeans=False,
                medianprops={'color': 'red'}, meanprops= {'color': 'orange'},
                patch_artist = True, boxprops = {'color':'grey','facecolor':'lightyellow'},
                showfliers=False)
    plt.show()
    plt.close()


def main():
    height_path = r'F:\ProductData\TEST\WSF3D_height'
    # height_path = r'D:\Experiments\Results\V5.1Buffer0_70\pred'
    story_path = r'D:\Experiments\Results\V5.1Buffer0_70\pred'
    lab_path = r'D:\Experiments\Results\V5.1Buffer0_70\lab'
    aim_heights = range(3, 61, 3)
    # aim_heights = [3, 6]
    eligible_heights = []
    story_results_all = []
    for aim_height in aim_heights:
        print(f'----------------processing height {aim_height}-----------------')
        height_results = get_height_dict_from_product(aim_height, height_path, lab_path)  # list of dict
        if len(height_results) == 0:
            continue
        else:
            eligible_heights.append(aim_height)
        print('----------------height_results complete-----------------')

        story_results = get_story_list_by_height_results(height_results, story_path)  # list of story value(int)
        print('----------------story_results complete-----------------')
        story_results_all.append(story_results)
    if len(story_results_all) == 0:
        print("No building is eligible")
        return
    print('----------------plotting results-----------------')
    plot_story_diffs(story_results_all, eligible_heights)


if __name__ == '__main__':
    # positions = [1, 2, 4, 7]
    # data = [
    #     np.random.normal(1, 5, 100).tolist(),
    #     np.random.normal(20, 5, 100).tolist(),
    #     np.random.normal(30, 5, 100).tolist(),
    #     np.random.normal(40, 5, 100).tolist(),
    # ]
    # plot_story_diffs(data, positions)
    main()