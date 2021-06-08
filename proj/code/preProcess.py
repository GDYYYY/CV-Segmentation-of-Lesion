import os
import numpy as np
import skimage.morphology as mplg
import skimage.measure as measure
import cv2


# input_path='../honeycombing_parenchyma'  stage 1 的结果地址
# output_path="../honeycombing_preprocess"  预处理输出地址
# input_mask_path="../pretreat/honeycombingMask"  stage 1 的mask地址
# open_radius=4
# close_radius=10
# label_size=2000   大联通级的最小面积
def preprocess(path_isReticular, isReticular, open_radius, close_radius, label_size):
    skip = False
    if path_isReticular:
        input_path = '../reticular_parenchyma'
        if isReticular:
            input_mask_path = "../pretreat/reticularMask/reticular"
            output_path = "../reticular_preprocess/reticular"
        else:
            skip = True
            input_mask_path = "../pretreat/reticularMask/reticular"
            output_path = "../reticular_preprocess/honeycombing"
    else:
        input_path = '../honeycombing_parenchyma'
        if isReticular:
            output_path = "../honeycombing_preprocess/reticular"
            input_mask_path = "../pretreat/honeycombingMask/reticular"
        else:
            output_path = "../honeycombing_preprocess/honeycombing"
            input_mask_path = "../pretreat/honeycombingMask/honeycombing"
    tmp = np.zeros((512, 512))
    if not skip:
        for root, dirs, files in os.walk(input_mask_path):
            for file in files:
                path = os.path.join(root, file)
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                tmp[gray > 1] = 1

    for root, dirs, files in os.walk(input_path):
        for file in files:
            path = os.path.join(root, file)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            low = 1
            high = np.max(gray)
            thresh = 0
            thresh_new = (int(low) + high) / 2
            while (thresh - thresh_new).__abs__() > 1:

                thresh = thresh_new
                row, col = gray.shape
                low_count = 0
                low_sum = 0
                high_count = 0
                high_sum = 0

                for i in range(row):
                    for j in range(col):
                        if gray[i, j] == 0:
                            continue
                        if gray[i, j] < thresh:
                            low_count += 1
                            low_sum += gray[i, j]
                        else:
                            high_count += 1
                            high_sum += gray[i, j]

                thresh_new = (low_sum / low_count + high_sum / high_count) / 2

            binary = np.zeros(gray.shape)
            for i in range(row):
                for j in range(col):
                    if 0 < gray[i, j] < thresh:
                        binary[i, j] = 1

            opened = mplg.opening(binary, mplg.disk(open_radius))
            labeled = measure.label(opened, connectivity=1)
            mask_sum = np.zeros(labeled.shape)
            count = len(np.unique(labeled))
            for i in range(count - 1):
                mask = np.zeros(labeled.shape)
                mask[labeled == i + 1] = 1
                size = len(mask[mask == 1])
                if size > label_size:
                    mask_sum += mask

            closed = mplg.closing(mask_sum, mplg.disk(close_radius))

            # triple = np.zeros(gray.shape)
            # for i in range(row):
            #     for j in range(col):
            #         if 0 < gray[i, j] < thresh:
            #             triple[i, j] = 1
            #         if gray[i, j] >= thresh:
            #             triple[i, j] = 2
            # output = closed * 255
            img[closed == 1] = (0, 0, 0)
            img[tmp == 0] = (0, 0, 0)
            # for i in range(row):
            #     for j in range(col):
            #         if 15 < gray[i, j] < thresh and closed[i, j] == 0:
            #             img[i, j] = (127, 127, 127)
            #         elif gray[i, j] > thresh and closed[i, j] == 0:
            #             img[i, j] = (255, 255, 255)
            cv2.imwrite(os.path.join(output_path, file), img)
            # # opened[closed == 1] = 0
            # cv2.imwrite(os.path.join(mask_path, file), opened * 255)


if __name__ == '__main__':
    print(3)
    preprocess(False, 4, 10, 2000)
