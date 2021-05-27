import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import re


def match(template, target):
    # result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
    # # 归一化处理
    # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    # # 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # strmin_val = str(min_val)
    # print(min_loc)
    # print(target.shape[:2])
    # return min_loc, strmin_val

    MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10
    sift = cv2.SIFT_create()  # 创建sift检测器
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)
    # 设置Flannde参数
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    # 舍弃大于0.7的匹配
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # matchesMask = mask.ravel().tolist()
        h, w = template.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        print(np.int32(dst))
        cv2.polylines(target, [np.int32(dst)], True, (255, 0, 0), 2, cv2.LINE_AA)
        return np.int32(dst)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    #     matchesMask = None
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=None,
    #                    matchesMask=matchesMask,
    #                    flags=2)
    # result = cv2.drawMatches(template, kp1, target, kp2, good, None, **draw_params)
    # plt.imshow(result, 'gray')
    # # plt.imshow(target)
    # plt.show()


def focalSegment(isReticular):
    # 对不同病灶的处理仅需修改参数isReticular，会相应修改路径、筛选颜色、大小等
    if isReticular:
        reticular_path = "../reticular"
        pretreat_path = "../pretreat/reticular"
        pretreat_mask_path = "../pretreat/reticularMask"
    else:
        reticular_path = "../honeycombing"
        pretreat_path = "../pretreat/honeycombing"
        pretreat_mask_path = "../pretreat/honeycombingMask"

    for root, dirs, files in os.walk(reticular_path):
        for filename in files:
            if "snapshot" in filename:
                path = os.path.join(root, filename)
                standard = cv2.imread(path)

                gray = cv2.cvtColor(standard, cv2.COLOR_BGR2GRAY)
                row_size, col_size = gray.shape
                # 去除左下角水印
                # print(col_size, row_size)
                for i in range(row_size - 150, row_size):
                    for j in range(200):
                        gray[i][j] = 0

                # 获取相应原图
                num = re.sub("\D", "", filename)
                originname = num.zfill(2) + "00001.jpg"
                originpath = os.path.join(reticular_path, originname)
                origin = cv2.imread(originpath)
                target = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)

                # 匹配缩放
                # match(gray, target)
                # continue

                if isReticular:
                    standard = cv2.resize(standard, (470, 346))  # reticular
                else:
                    standard = cv2.resize(standard, (529, 391))  # honeycombing

                hsv = cv2.cvtColor(standard, cv2.COLOR_BGR2HSV)
                if isReticular:
                    # 提取黄色
                    low_hsv = np.array([26, 43, 46])
                    high_hsv = np.array([34, 255, 255])
                else:
                    # 提取紫色
                    low_hsv = np.array([125, 43, 46])
                    high_hsv = np.array([155, 255, 255])
                mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
                # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                # 去除左下角水印
                # print(col_size, row_size)
                small_row_size, small_col_size = mask.shape
                for i in range(small_row_size - 47, small_row_size):
                    for j in range(63):
                        mask[i][j] = 0
                # 新建与origin对齐的mask
                row_size, col_size = origin.shape[:2]
                newmask = np.zeros(origin.shape, dtype=np.uint8)

                small_col_size = min(small_col_size, col_size)
                for i in range(small_row_size - 1):
                    for j in range(small_col_size - 1):
                        if isReticular:
                            newmask[i + 82][j + 20] = mask[i][j]  # reticular
                        else:
                            newmask[i + 60][j] = mask[i][j + 9]  # honeycombing

                cv2.imwrite(os.path.join(pretreat_mask_path, originname), newmask)

                # 匹配原图
                # standard[mask == 0] = (0, 0, 0)
                # gray = cv2.cvtColor(standard, cv2.COLOR_BGR2GRAY)
                # cv2.imwrite(os.path.join(pretreat_path, originname), gray)
                origin[newmask == 0] = 0
                cv2.imwrite(os.path.join(pretreat_path, originname), origin)


if __name__ == '__main__':
    focalSegment(isReticular=True)
