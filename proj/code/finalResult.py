import os
import numpy as np
import skimage.morphology as mplg
import skimage.measure as measure
import cv2
import scipy.ndimage as ndimage

answer_path = "../pretreat/reticular"
result_path = "../reticular_result/entropy"
final_path = "../reticular_result/final"
origin_path = "../reticular"


def finalResult():
    sum = 0
    for root, dirs, files in os.walk(result_path):
        for filename in files:
            path = os.path.join(root, filename)
            result = cv2.imread(path)
            answer = cv2.imread(os.path.join(answer_path, filename))
            answer_gray = cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            origin = cv2.imread(os.path.join(origin_path, filename))

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
            sum += ret
            print(ret)

            # 上色
            for i in range(row):
                for j in range(col):
                    if result_gray[i][j] > 0:
                        origin[i][j] = (144, 238, 144)
            cv2.imwrite(os.path.join(final_path, filename), origin)
    print("average:" + str(sum / 20))


if __name__ == '__main__':
    finalResult()
