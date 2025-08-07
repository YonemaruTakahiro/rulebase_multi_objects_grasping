from __future__ import print_function
import os
import sys
import cv2
import numpy as np
import argparse
import random as rng

rng.seed(12345)

# The result turned out to be very bad
def thresh_callback(val):
    threshold = val

    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours): 
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])


    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)): 
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        # cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    cv2.imshow('Contours', drawing)

def drawContours(img):
    ret, thresh = cv2.threshold(img, 100, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours)): 
        cnt = contours[i]
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(img, [cnt], 0, color, 3)

    cv2.imshow('Contours', img)


parser = argparse.ArgumentParser(description='Code for creating buonding box and circles for contours tutorial')
parser.add_argument('--input', help='Path to input image', default='stuff.jpg')
args = parser.parse_args()


src = cv2.imread('test.png', 0)
#src = cv2.resize(src, (580, 622))

# Convert image to gray and blur it
# src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

drawContours(src)



cv2.waitKey()