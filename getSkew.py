import cv2
import numpy as np
import math

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

# dilation
kernel = np.ones((7, 13), np.uint8)
dilated = cv2.dilate(thres, kernel, iterations=1)
cv2.imshow('dilated', dilated)
cv2.waitKey(0)

# erode
kernel = np.ones((8, 1), np.uint8)
eroded = cv2.erode(dilated, kernel, iterations=1)
cv2.imshow('eroded', eroded)
cv2.waitKey(0)

# find text
im2, ctrs, heir = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# delete small blobs from image
n_ctrs = []
for ctr in ctrs:
    area = cv2.contourArea(ctr)
    if area >= 190:
        n_ctrs.append(ctr)
del ctrs

# sorting contours
n_ctrs = sorted(n_ctrs, key=lambda x: cv2.boundingRect(x))

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


# detecting slope for line
fit = np.polyfit(c_x, c_y, deg=1)

# line points
pt1 = (c_x[0], int(c_x[0] * fit[0] + fit[1]))
pt2 = (c_x[-1], int(c_x[-1] * fit[0] + fit[1]))
pt3 = (c_x[-1], int(c_x[0] * fit[0] + fit[1]))

# fitted line
cv2.line(image, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

# hor-axis line
cv2.line(image, pt1, pt3, (0, 255, 0), 2, cv2.LINE_AA)

# mainslope print
skew = fit[0]
print(skew)


cv2.imshow('final', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
