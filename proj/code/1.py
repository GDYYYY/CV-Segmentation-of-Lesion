import os
import numpy as np
import skimage.morphology as mplg
import skimage.measure as measure
import cv2
import scipy.ndimage as ndimage


def function(isHoneycombing):
    reticular_path = "../reticular"
    honeycombing_path = '../honeycombing'
    reticular_mask_path = "../reticular_mask"
    reticular_parenchyma_path = "../reticular_parenchyma"
    honeycombing_mask_path = "../honeycombing_mask"
    honeycombing_parenchyma_path = "../honeycombing_parenchyma"

    if isHoneycombing:
        input_path = honeycombing_path
        mask_path = honeycombing_mask_path
        parenchyma_path = honeycombing_parenchyma_path
    else:
        input_path = reticular_path
        mask_path = reticular_mask_path
        parenchyma_path = reticular_parenchyma_path

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if "snapshot" in file:
                continue
            path = os.path.join(root, file)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # cv2.imwrite(os.path.join("../tmp", file), gray)

            low = np.min(gray)
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
                        if gray[i, j] < thresh:
                            low_count += 1
                            low_sum += gray[i, j]
                        else:
                            high_count += 1
                            high_sum += gray[i, j]

                thresh_new = (low_sum / low_count + high_sum / high_count) / 2

            binary = np.where(gray < thresh, 1, 0)

            # cv2.imwrite(os.path.join("../tmp", file), binary*255)
            row, col = gray.shape
            mask = np.zeros((row + 2, col + 2))
            filled = np.uint8(binary)
            cv2.floodFill(filled, np.uint8(mask), (0, 0), 0)
            cv2.floodFill(filled, np.uint8(mask), (col - 1, row - 1), 0)
            cv2.floodFill(filled, np.uint8(mask), (0, row - 1), 0)
            cv2.floodFill(filled, np.uint8(mask), (col - 1, 0), 0)

            # cv2.imwrite(os.path.join("../tmp", file), filled * 255)

            labeled = measure.label(filled)
            closed = np.zeros(labeled.shape)
            count = len(np.unique(labeled))
            for i in range(count):
                mask = np.zeros(labeled.shape)
                mask[labeled == i + 1] = 1
                size = len(mask[mask == 1])
                if size < 600 or size > 700:
                    closed += mask
            # closed = mplg.closing(binary, np.ones((2, 2)))

            # cv2.imwrite(os.path.join("../tmp", file), closed * 255)

            closed = mplg.closing(closed, mplg.disk(5))

            # cv2.imwrite(os.path.join("../tmp", file), closed * 255)

            # closed = mplg.erosion(closed, np.ones((3, 3)))
            # closed = mplg.dilation(closed, np.ones((9, 9)))
            objected = mplg.remove_small_objects(closed.astype(bool), 500)

            # cv2.imwrite(os.path.join("../tmp", file), objected * 255)

            opened = mplg.opening(objected, mplg.disk(5))

            # cv2.imwrite(os.path.join("../tmp", file), opened * 255)

            closed = mplg.closing(opened, mplg.disk(10))

            # cv2.imwrite(os.path.join("../tmp", file), closed * 255)

            output = closed * 255
            cv2.imwrite(os.path.join(mask_path, file), output)

            img[closed == 0] = (0, 0, 0)
            cv2.imwrite(os.path.join(parenchyma_path, file), img)


if __name__ == '__main__':
    function(True)
    function(False)
