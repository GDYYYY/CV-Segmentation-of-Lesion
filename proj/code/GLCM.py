import os
import cv2
import matplotlib.pyplot as plt


def focalSegment():
    reticular_path = "../reticular"
    parenchyma_path = "../parenchyma"

    for root, dirs, files in os.walk(parenchyma_path):
        for filename in files:
            # 读取肺实质图像
            path = os.path.join(root, filename)
            maskedImg = cv2.imread(path)
            # 读取原始图像
            imgPath = os.path.join(reticular_path, filename)
            img = cv2.imread(imgPath)
            # 生成灰度图用于处理
            gray = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2GRAY)
            row_size, col_size = gray.shape

            # 生成直方图
            plot = list(filter(lambda a: a != 0, gray.flatten()))
            plt.hist(plot, bins=20)
            plt.show()

            # 输出最终病灶结果
            cv2.imwrite(r"../focalResults/" + filename, img)


if __name__ == '__main__':
    focalSegment()
