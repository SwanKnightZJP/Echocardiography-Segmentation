import cv2 as cv
import numpy as np


def sobel(img):
    sobel_edge_x = cv.Sobel(img, ddepth=cv.CV_32F, dx=1, dy=0, ksize=5)
    sobel_edge_x = np.abs(sobel_edge_x)
    sobel_edge_x = sobel_edge_x/(np.max(sobel_edge_x)+0.000000000000000001)
    sobel_edge_x = sobel_edge_x*255  #进行归一化处理
    sobel_edge_x = sobel_edge_x.astype(np.uint8)

    sobel_edge_y = cv.Sobel(img, ddepth=cv.CV_32F, dx=0, dy=1, ksize=5)
    sobel_edge_y = np.abs(sobel_edge_y)
    sobel_edge_y = sobel_edge_y/(np.max(sobel_edge_y)+0.000000000000000001)
    sobel_edge_y = sobel_edge_y*255
    sobel_edge_y = sobel_edge_y.astype(np.uint8)

    sobel_edge1 = cv.addWeighted(sobel_edge_x, 0.5, sobel_edge_y, 0.5, 0)

    sobel_edge = cv.Sobel(img, ddepth=cv.CV_32F, dx=1, dy=1, ksize=5)
    sobel_edge = np.abs(sobel_edge)
    sobel_edge = sobel_edge/(np.max(sobel_edge)+0.000000000000000001)
    sobel_edge = sobel_edge*255
    sobel_edge = sobel_edge.astype(np.uint8)

    # post precess
    # sobel_edge1[sobel_edge1 > 10] = 255

    # #
    sobel_edge1 = sobel_edge1.astype(np.float) - 10
    sobel_edge1[sobel_edge1 < 0] = 0
    sobel_edge1 = sobel_edge1 * 6.375
    sobel_edge1[sobel_edge1 > 255] = 255
    sobel_edge1 = sobel_edge1.astype(np.uint8)

    return sobel_edge_x, sobel_edge_y, sobel_edge, sobel_edge1