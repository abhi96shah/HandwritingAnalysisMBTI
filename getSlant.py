import cv2
import numpy as np
import matplotlib.pyplot as plt


def findContours(dst):

    # dilation
    kernel = np.ones((9, 30), np.uint8)
    img_dilation = cv2.dilate(dst, kernel, iterations=1)
    # cv2.imshow("dilated", img_dilation)
    # cv2.waitKey(0)
    # find contours
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse=False)

    return sorted_ctrs


def getSlant(image):
    cols = np.zeros((1, len(image[0])), np.uint8)
    col_count = 0
    for col_count in range(len(image[0])):
        for c in image:
            if c[col_count] == 255:
                cols[0][col_count] += 1

    plt.plot(range(0, len(cols[0])), cols[0])
    plt.show()


def doShear(image):


image = cv2.imread("DATA/002_Abhishek.jpg")


image = cv2.resize(image, (0, 0), fx=1.2, fy=1.2)
# cv2.imshow('orignal', image)
# cv2.waitKey(0)

image = image[30:image.shape[0] - 30, 30:image.shape[1] - 30]

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)

# binary
ret, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('second', thresh)
# cv2.waitKey(0)

# dilation
kernel = np.ones((3, 5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imshow("dilated1", img_dilation)
# cv2.waitKey(0)

# Open to de-noise
kernel_open = np.ones((5, 5), np.uint8)
img_open = cv2.morphologyEx(img_dilation, cv2.MORPH_OPEN, kernel_open, iterations=1)
# cv2.imshow("Open image", img_open)
# cv2.waitKey(0)

sorted_ctrs = findContours(img_open)


#<------EXTRACTING FEATURES------>
sum_height = 0
sum_width = 0
count = 0
c_x = []
c_y = []
Leftmost = 0
Rightmost = 0
sum_dist = 0
sum_int = 0

for i, ctr in enumerate(sorted_ctrs):
    if (cv2.contourArea(ctr) > 1200.0):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        c_x.append(int(x + (w / 2)))
        c_y.append(int(y + (h / 2)))

        # Getting ROI
        roi = img_open[y:y + h, x:x + w]
        # show ROI
        cv2.imshow('segment no:' + str(i), roi)
        cv2.waitKey(0)
        # print(sum_int)
        # cv2.imshow("After changing ", roi)
        # cv2.waitKey(0)

        # t Creating ROI and Center on Image
        cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
        cv2.circle(image, (c_x[-1], c_y[-1]), 0, (255, 0, 0), 5)

        # Draw every contour and rectangle
        # cv2.drawContours(image, ctr, -1, (255,2,0), 3)
        # cv2.waitKey(0)
        getSlant(roi)
        doShear(roi)
