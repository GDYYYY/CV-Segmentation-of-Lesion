import os
import numpy as np
import skimage.morphology as mplg
import skimage.measure as measure
import cv2
import scipy.ndimage as ndimage

reticular_path = "../reticular_parenchyma"
honeycombing_path = '../honeycombing_parenchyma'
reticular_mask_path = "../reticular_preprocess_mask"
reticular_preprocess_path = "../reticular_preprocess"
honeycombing_mask_path = "../honeycombing_preprocess_mask"
honeycombing_preprocess_path = "../honeycombing_preprocess"

reticular = "reticular"
honeycombing = "honeycombing"

type = reticular

if type == honeycombing:
    input_path = honeycombing_path
    mask_path = honeycombing_mask_path
    preprocess_path = honeycombing_preprocess_path
else:
    input_path = reticular_path
    mask_path = reticular_mask_path
    preprocess_path = reticular_preprocess_path

tmp = np.zeros((512, 512))
for root, dirs, files in os.walk("../pretreat/reticularMask"):
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

        opened = mplg.opening(binary, mplg.disk(4))
        labeled = measure.label(opened, connectivity=1)
        mask_sum = np.zeros(labeled.shape)
        count = len(np.unique(labeled))
        for i in range(count - 1):
            mask = np.zeros(labeled.shape)
            mask[labeled == i + 1] = 1
            size = len(mask[mask == 1])
            if size > 250:
                mask_sum += mask

        closed = mplg.closing(mask_sum, mplg.disk(5))
        img[closed == 1] = (0, 0, 0)
        img[tmp == 0] = (0, 0, 0)
        # triple = np.zeros(gray.shape)
        # for i in range(row):
        #     for j in range(col):
        #         if 0 < gray[i, j] < thresh:
        #             triple[i, j] = 1
        #         if gray[i, j] >= thresh:
        #             triple[i, j] = 2
        # output = closed * 255

        cv2.imwrite(os.path.join(preprocess_path, file), img)
