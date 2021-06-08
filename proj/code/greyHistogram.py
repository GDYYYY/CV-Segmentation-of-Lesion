import os
import cv2
import matplotlib.pyplot as plt


def focalSegment(isReticular):
    reticular_path = "../pretreat/reticular"
    honeycombing_path = "../pretreat/honeycombing"
    origin_path = "../honeycombing_parenchyma"
    if isReticular:
        mask_path = reticular_path
        origin_path = "../reticular_result/entropy"
        result_path = "../reticular_result/final"
    else:
        mask_path = honeycombing_path
        origin_path = "../honeycombing_parenchyma"
        result_path = "../honeycombing_result/"

    for root, dirs, files in os.walk(mask_path):
        for filename in files:
            # 读取病灶图
            imgPath = os.path.join(mask_path, filename)
            img = cv2.imread(imgPath)
            # 读取原图
            originPath = os.path.join(origin_path, filename)
            origin = cv2.imread(originPath)
            # 生成灰度图用于处理
            gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
            row_size, col_size = gray.shape

            # # 生成直方图
            # plot = list(filter(lambda a: a != 0, gray.flatten()))
            # plt.hist(plot, bins=20)
            # plt.show()

            if isReticular:
                low, high = (100, 255)
            else:
                low, high = (200, 255)

            for i in range(row_size):
                for j in range(col_size):
                    if low < gray[i][j] < high:
                        img[i][j] = (144, 238, 144)

            # 输出最终病灶结果
            cv2.imwrite(os.path.join(result_path, filename), img)


if __name__ == '__main__':
    focalSegment(isReticular=True)
