import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import skimage.morphology as mplg
import skimage.measure as measure

typeString = ['contrast', 'energy', 'entropy', 'homogeneity']


# gray_level = 16  # 灰度级数
# window_w, window_h = 12, 12  # 窗口大小
# dx, dy = 1, 1  # 灰度共生矩阵计算方向
#
# featureType = 3  # 对应typeString选择计算的特征值
# low, high = 0, 80  # 筛选区域，保留特征值在(low,high)之间的小窗口


# def pre(number):
#     global window_w, window_h
#     # window_w, window_h = window[number], window[number]
#     # global low
#     # low = lows[number]


def getMaxGrayLevel(img):
    max_gray_level = 0
    height, width = img.shape[:2]
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def grayDown(img, max_gray_level, gray_level):
    height, width = img.shape[:2]
    # 减小灰度级数
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                img[j][i] = img[j][i] * gray_level / max_gray_level
    return img


def getGLCM(img, dx, dy, gray_level):
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


def feature(m, gray_level):
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


def checkWindow(w, threshold):
    height, width = w.shape[:2]
    for i in range(height):
        for j in range(width):
            if w[i][j] < threshold:
                return False
    return True


def computeGray(w):
    height, width = w.shape[:2]
    sum = 0
    for i in range(height):
        for j in range(width):
            sum += w[i][j]
    return sum


def focalSegment(path_isReticular, isReticular):
    if path_isReticular:
        if isReticular:
            result_path = "../reticular_result/reticular"
            parenchyma_path = "../reticular_preprocess/reticular"
        else:
            result_path = "../reticular_result/honeycombing"
            parenchyma_path = "../reticular_preprocess/honeycombing"
    else:
        if isReticular:
            result_path = "../honeycombing_result/reticular"
            parenchyma_path = "../honeycombing_preprocess/reticular"
        else:
            result_path = "../honeycombing_result/honeycombing"
            parenchyma_path = "../honeycombing_preprocess/honeycombing"

    if isReticular:
        gray_level = 16  # 灰度级数
        window_w, window_h = 5, 5  # 窗口大小
        dx, dy = 0, 1  # 灰度共生矩阵计算方向
        featureType = 2  # 对应typeString选择计算的特征值
        low, high = -11, 0  # 筛选区域，保留特征值在(low,high)之间的小窗口
        k = 7
    else:
        gray_level = 16  # 灰度级数
        window_w, window_h = 12, 12  # 窗口大小
        dx, dy = 1, 1  # 灰度共生矩阵计算方向
        featureType = 3  # 对应typeString选择计算的特征值
        low, high = 0, 80  # 筛选区域，保留特征值在(low,high)之间的小窗口
        k = 24

    for root, dirs, files in os.walk(parenchyma_path):  # 选择计算灰度共生矩阵的图
        for filename in files:
            # 读取肺实质图像
            path = os.path.join(root, filename)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            max_gray = getMaxGrayLevel(gray)
            gray = grayDown(gray, max_gray, gray_level)  # 减少灰度级数

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
                        glcm = getGLCM(window, dx, dy, gray_level)

                        featureValue = feature(glcm, gray_level)
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

            # # 筛选
            # for i in range(height):
            #     for j in range(width):
            #         if compareImg[i][j] <= low or compareImg[i][j] >= high:
            #             img[i][j] = 0
            mask = np.zeros(gray.shape[:2], dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    if low < compareImg[i][j] < high:
                        mask[i][j] = 255
            # 填空隙
            # mask = mplg.opening(mask, mplg.disk(2))
            mask = mplg.closing(mask, mplg.disk(k))
            # mask = mplg.opening(mask, mplg.disk((window_h+window_w)/4))

            if isReticular:
                for i in range(height):
                    for j in range(width):
                        if checkWindow(gray[i:i + 6, j:j + 6], 14):
                            mask[i:i + 6, j:j + 6] = 0

            img[mask == 0] = 0
            # 输出最终病灶结果
            cv2.imwrite(os.path.join(result_path, filename), img)


#
# answer_path = "../pretreat/honeycombing"
# result_path = "../honeycombing_result/" + typeString[featureType]
# final_path = "../honeycombing_result/final"
# origin_path = "../honeycombing"


def IOUResult(path_isReticular, isReticular):
    if path_isReticular:
        if isReticular:
            answer_path = "../pretreat/reticular"
            result_path = "../reticular_result/reticular"
        else:
            answer_path = ""
            result_path = ""

    else:
        if isReticular:
            answer_path = ""
            result_path = ""
        else:
            answer_path = "../pretreat/honeycombing"
            result_path = "../honeycombing_result/honeycombing"

    avg = 0
    for root, dirs, files in os.walk(result_path):
        for filename in files:
            path = os.path.join(root, filename)
            result = cv2.imread(path)
            answer = cv2.imread(os.path.join(answer_path, filename))
            answer_gray = cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            # IOU
            intersection = 0
            union = 0
            row, col = result.shape[:2]
            for i in range(row):
                for j in range(col):
                    if answer_gray[i][j] > 0 and result_gray[i][j] > 0:
                        intersection += 1
                        union += 1
                    elif answer_gray[i][j] > 0 or result_gray[i][j] > 0:
                        union += 1
            ret = float(intersection) / float(union)
            print(filename, ' ', ret)
            avg += ret / 20

    print(" avg:", avg)


def color(path_isReticular, isReticular):
    if path_isReticular:
        final_path = "../reticular_result/final"
        if isReticular:
            result_path = "../reticular_result/reticular"
            origin_path = "../reticular"
        else:
            result_path = "../reticular_result/honeycombing"
            origin_path = "../reticular_result/final"

    else:
        final_path = "../honeycombing_result/final"
        if isReticular:
            result_path = "../honeycombing_result/reticular"
            origin_path = "../honeycombing_result/final"
        else:
            result_path = "../honeycombing_result/honeycombing"
            origin_path = "../honeycombing"

    for root, dirs, files in os.walk(result_path):
        for filename in files:
            path = os.path.join(root, filename)
            result = cv2.imread(path)
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            origin = cv2.imread(os.path.join(origin_path, filename))
            row, col = result.shape[:2]
            # 上色
            for i in range(row):
                for j in range(col):
                    if final_path == origin_path and np.abs(int(origin[i, j, 1]) - int(origin[i, j, 2])) > 50:
                        continue
                    if result_gray[i][j] > 5:
                        if isReticular:
                            if origin[i, j, 2] > 185:
                                origin[i, j] = (185, 185, 255)
                            else:
                                origin[i][j][2] += 70
                        else:
                            if origin[i, j, 1] > 185:
                                origin[i, j] = (185, 255, 185)
                            else:
                                origin[i][j][1] += 70
            cv2.imwrite(os.path.join(final_path, filename), origin)


if __name__ == '__main__':
    focalSegment(False, False)
    IOUResult(False, False)
    color(False, False)
