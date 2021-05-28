import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

gray_level = 16  # 灰度级数
window_w=7
window_h=7

def getMaxGrayLevel(img):
    max_gray_level = 0
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGLCM(img, dx, dy):
    ans = np.zeros(img.shape, dtype=np.uint8)
    max_gray_level = getMaxGrayLevel(img)
    height, width = img.shape
    # 减小灰度级数
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                img[j][i] = img[j][i] * gray_level / max_gray_level

    # 计算共生矩阵
    for j in range(height - dy):
        for i in range(width - dx):
            x = img[j][i]
            y = img[j + dy][i + dx]
            ans[x][y] += 1
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

    return contrast, energy, -entropy, homogeneity


def focalSegment():
    reticular_path = "../reticular"
    parenchyma_path = "../reticular_parenchyma"

    for root, dirs, files in os.walk(parenchyma_path):
        for filename in files:
            # 读取肺实质图像
            path = os.path.join(root, filename)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            glcm_0 = getGLCM(gray, 1, 0)
            # print(glcm_0)
            contrast, energy, entropy, homogeneity = feature(glcm_0)
            print(contrast, energy, entropy, homogeneity)

            # 输出最终病灶结果
            # cv2.imwrite(r"../focalResults/" + filename, img)


if __name__ == '__main__':
    focalSegment()
