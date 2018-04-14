import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage import transform

# import image
image = cv2.imread("./DATA/Baseline.jpg", cv2.IMREAD_COLOR)
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('image', image)
cv2.waitKey(0)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)


# threshold image
ret, thres = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('threshold', thres)
cv2.waitKey(0)

# erode
kernel = np.ones((3, 1), np.uint8)
eroded = cv2.erode(thres, kernel, iterations=1)
cv2.imshow('eroded', eroded)
cv2.waitKey(0)

# dilation
kernel = np.ones((5, 9), np.uint8)
dilated = cv2.dilate(eroded, kernel, iterations=1)
cv2.imshow('dilated', dilated)
cv2.waitKey(0)


# find text
im2, ctrs, heir = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# remove small contours
n_ctrs = []
for ctr in ctrs:
    area = cv2.contourArea(ctr)
    if area >= 190:
        n_ctrs.append(ctr)
del ctrs

# draw rectangles around text and center points
c_x = []
c_y = []
for i in range(len(n_ctrs)):
    x, y, w, h = cv2.boundingRect(n_ctrs[i])
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)
    c_x.append(int(x + w / 2))
    c_y.append(int(y + h / 2))

for i in range(len(c_x)):
    cv2.circle(image, (c_x[i], c_y[i]), 0, (255, 0, 0), 5)

cv2.imshow('final', image)
cv2.waitKey(0)

# for each contour


def testVFreq(image):

    bins = [0] * image.shape[1]
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            if image[j, i] == 255:
                bins[i] += 1

    afine_tf = transform.AffineTransform(shear=0.5)
    # Apply transform to image data
    modified = transform.warp(image, inverse_map=afine_tf)

    cv2.imshow("test", image)
    cv2.waitKey(0)
    plt.plot(bins)
    plt.subplot(211)
    print(bins)
    print(type(modified))

    bins1 = [0] * modified.shape[1]
    for i in range(modified.shape[1]):
        for j in range(modified.shape[0]):
            if modified[j, i] == 255:
                bins1[i] += 1

    cv2.imshow("test1", modified)
    cv2.waitKey(0)
    plt.plot(bins1)
    plt.subplot(212)
    print(bins1)
    plt.show()


for i in range(len(n_ctrs)):
    x, y, w, h = cv2.boundingRect(n_ctrs[i])
    roi = thres[y:y + h, x:x + w]
    # cv2.imshow('roitest' + str(i), roi)
    # cv2.waitKey(0)
    # print(image.shape)
testVFreq(roi)

# cv2.imshow('roi', image)
# cv2.waitKey(0)
cv2.destroyAllWindows()
