import os
import numpy as np
import skimage.morphology as mplg
import skimage.measure as measure
import cv2
import scipy.ndimage as ndimage

answer_path = "../pretreat/honeycombing"
result_path = "../honeycombing_result/contrast"

answer = cv2.imread(answer_path)
answer_gray = cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)
result = cv2.imread(result_path)
result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

intersection = 0
union = 0
row, col = result.shape
for i in range(row):
    for j in range(col):
        if answer_gray[i, j] > 0 and result_gray[i, j] > 0:
            intersection += 1
            union += 1
        elif answer_gray[i, j] > 0 or result_gray[i, j] > 0:
            union += 1
ret = float(intersection) / float(union)
print(ret)
