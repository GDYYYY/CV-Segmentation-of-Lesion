import os
import numpy as np
import skimage.morphology as mplg
import skimage.measure as measure
import cv2
import scipy.ndimage as ndimage

from preProcess import preprocess
import GLCM
import pretreatMask



def finalResult():
    # for reticular 针对该文件夹内图像
    print('for files in reticular')
    # 获取标准病灶掩膜
    pretreatMask.focalSegment(isReticular=True)
    # 对图像进行预处理
    preprocess(True, True, 4, 5, 300)
    GLCM.focalSegment(path_isReticular=True, isReticular=True)  # 算reticular
    GLCM.IOUResult(path_isReticular=True, isReticular=True)  # 算分
    GLCM.color(path_isReticular=True, isReticular=True)  # 上色
    preprocess(True, False, 4, 5, 300)
    GLCM.focalSegment(path_isReticular=True, isReticular=False)  # 算蜂巢
    GLCM.color(path_isReticular=True, isReticular=False)  # 上色

    # for honeycombing
    print('for files in honeycombing')
    pretreatMask.focalSegment(isReticular=False)
    preprocess(False, False, 4, 10, 2000)
    GLCM.focalSegment(path_isReticular=False, isReticular=False)  # 算蜂巢
    GLCM.IOUResult(path_isReticular=False, isReticular=False)  # 算分
    GLCM.color(path_isReticular=False, isReticular=False)  # 上色
    preprocess(False, True, 4, 10, 2000)
    GLCM.focalSegment(path_isReticular=False, isReticular=True)  # 算reticular
    GLCM.color(path_isReticular=False, isReticular=True)  # 上色


if __name__ == '__main__':
    finalResult()
