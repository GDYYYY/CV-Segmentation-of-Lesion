import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import skimage.morphology as mplg

gray_level = 16  # 灰度级数
typeString = ['contrast', 'energy', 'entropy', 'homogeneity']

window_w, window_h = 7, 7  # 窗口大小
dx, dy = 2, 2  # 灰度共生矩阵计算方向

featureType = 3  # 对应typeString选择计算的特征值
low, high = 0,10  # 筛选区域，保留特征值在(low,high)之间的小窗口


def getMaxGrayLevel(img):
    max_gray_level = 0
    height, width = img.shape[:2]
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def grayDown(img, max_gray_level):
    height, width = img.shape[:2]
    # 减小灰度级数
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                img[j][i] = img[j][i] * gray_level / max_gray_level
    return img


def getGLCM(img, dx, dy):
    ans = np.zeros((gray_level, gray_level), dtype=int)
    height, width = img.shape[:2]

    # 计算共生矩阵
    for j in range(height - dy):
        for i in range(width - dx):
            x = img[j][i]
            y = img[j + dy][i + dx]
            # print(j, i, x, y)
            ans[x][y] += 1
    # print('GLCM')
    # print(ans)

    # 归一化
    # sum = (height - dy) * (width - dx)
    # for i in range(gray_level):
    #     for j in range(gray_level):
    #         ans[i][j] = float(ans[i][j] / sum)

    return ans


def feature(m):
    contrast = 0.0  # 对比度
    energy = 0.0  # 能量
    entropy = 0.0  # 熵
    homogeneity = 0.0  # 一致性
    for i in range(gray_level):
        for j in range(gray_level):
            contrast += (i - j) * (i - j) * m[i][j]
            energy += int(m[i][j] * m[i][j])
            homogeneity += m[i][j] / (1 + (i - j) * (i - j))
            if m[i][j] > 0.0:
                entropy += m[i][j] * math.log(m[i][j])
    return [contrast, energy, -entropy, homogeneity]


def feature2gray(m, maxValue):
    print(maxValue)
    # if maxValue <= 255:
    #     return m
    height, width = m.shape[:2]
    for j in range(height):
        for i in range(width):
            m[j][i] = m[j][i] * 255 / maxValue
    return m


def focalSegment():
    origin_path = "../reticular_parenchyma"
    standard_path = "../pretreat/reticular"  # 标准答案
    parenchyma_path = "../reticular_preprocess"
    result_path = "../reticular_result/" + typeString[featureType]

    for root, dirs, files in os.walk(parenchyma_path):  # 选择计算灰度共生矩阵的图
        for filename in files:
            # 读取肺实质图像
            path = os.path.join(root, filename)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            max_gray = getMaxGrayLevel(gray)
            gray = grayDown(gray, max_gray)  # 减少灰度级数

            # 记录各特征值
            # cont = np.zeros(gray.shape[:2], dtype=int)
            # nrg = np.zeros(gray.shape[:2], dtype=int)
            # ntp = np.zeros(gray.shape[:2], dtype=int)
            # hom = np.zeros(gray.shape[:2], dtype=int)
            data = np.zeros(gray.shape[:2], dtype=int)

            # 按小区域计算GLCM
            hw = int(window_w / 2)
            hh = int(window_h / 2)
            for i in range(window_h, height - window_h, window_w):
                for j in range(window_w, width - window_w, window_h):
                    if gray[i][j] != 0:  # 提高计算效率
                        window = gray[i - hh:i + hh + 1, j - hw:j + hw + 1]
                        glcm = getGLCM(window, dx, dy)

                        featureValue = feature(glcm)
                        # contrast, energy, entropy, homogeneity = feature(glcm)
                        # cont[i - hh:i + hh + 1, j - hw:j + hw + 1] = contrast
                        # nrg[i - hh:i + hh + 1, j - hw:j + hw + 1] = energy
                        # ntp[i - hh:i + hh + 1, j - hw:j + hw + 1] = entropy
                        # hom[i - hh:i + hh + 1, j - hw:j + hw + 1] = homogeneity
                        data[i - hh:i + hh + 1, j - hw:j + hw + 1] = featureValue[featureType]

            compareImg = data

            # # 生成直方图
            # plot = list(filter(lambda a: a != 0, compareImg.flatten()))
            # plt.hist(plot, bins=20)
            # plt.show()

            # 筛选
            mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    if low < compareImg[i][j] < high:
                        mask[i][j] = 255
            # 填空隙
            mask = mplg.closing(mask, np.ones((window_h + 1, window_w + 1)))
            # mask = mplg.opening(mask, np.ones(( window_h, window_w)))
            img[mask == 0] = 0

            # 输出最终病灶结果
            cv2.imwrite(os.path.join(result_path, filename), img)


if __name__ == '__main__':
    focalSegment()
