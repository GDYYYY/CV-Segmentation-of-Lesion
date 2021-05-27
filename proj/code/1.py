import os
import numpy as np
import skimage.morphology as mplg
import skimage.measure as measure
import cv2
import scipy.ndimage as ndimage


def function():
    reticular_path = "../reticular"
    honeycombing_path = '../honeycombing'
    reticular_mask_path = "../reticular_mask"
    reticular_parenchyma_path = "../reticular_parenchyma"
    honeycombing_mask_path = "../honeycombing_mask"
    honeycombing_parenchyma_path = "../honeycombing_parenchyma"

    reticular = "reticular"
    honeycombing = "honeycombing"
    type = reticular

    if type == honeycombing:
        input_path = honeycombing_path
        mask_path = honeycombing_mask_path
        parenchyma_path = honeycombing_parenchyma_path
    else:
        input_path = reticular_path
        mask_path = reticular_mask_path
        parenchyma_path = reticular_parenchyma_path

    for root, dirs, files in os.walk(input_path):
        for file in files:
            path = os.path.join(root, file)
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

            # hole = ndimage.binary_fill_holes(binary)
            labeled = measure.label(binary)
            closed = np.zeros(labeled.shape)
            count = len(np.unique(labeled))
            for i in range(count):
                mask = np.zeros(labeled.shape)
                mask[labeled == i + 1] = 1
                size = len(mask[mask == 1])
                if size < 600 or size > 700:
                    closed += mask
            # closed = mplg.closing(binary, np.ones((2, 2)))
            # closed = mplg.remove_small_objects(closed.astype(bool), 2000)
            closed = mplg.closing(closed, np.ones((15, 15)))
            closed = mplg.opening(closed, np.ones((5, 5)))

            row, col = gray.shape
            mask = np.zeros((row + 2, col + 2))
            filled = np.uint8(closed)
            cv2.floodFill(filled, np.uint8(mask), (0, 0), 0)
            cv2.floodFill(filled, np.uint8(mask), (row - 1, col - 1), 0)

            # dilated = mplg.dilation(filled, np.ones((40, 40)))
            # labeled = mplg.label(dilated)
            # mask_sum = np.zeros(labeled.shape)
            # count = len(np.unique(labeled))
            # for i in range(count):
            #     mask = np.zeros(labeled.shape)
            #     mask[labeled == i + 1] = 1
            #     mask = mplg.closing(mask, np.ones((25, 25)))
            #     mask = ndimage.binary_fill_holes(mask)
            #     mask_sum += mask
            # mask_sum = mplg.erosion(mask_sum, np.ones((40, 40)))

            # opened = mplg.erosion(filled, np.ones((55, 55)))

            # hole = ndimage.binary_fill_holes(opened)
            # hole = mplg.remove_small_objects(opened, 10000)

            # binary = binary*255
            # cv2.imwrite(os.path.join(out_path, file), binary)
            output = filled * 255
            cv2.imwrite(os.path.join(mask_path, file), output)

            img[filled == 0] = (0, 0, 0)
            cv2.imwrite(os.path.join(parenchyma_path, file), img)


if __name__ == '__main__':
    function()
