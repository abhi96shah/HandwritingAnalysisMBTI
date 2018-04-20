from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn import datasets
import cv2
import numpy as np


itemList = list()


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

#<------BASELINE FIT------>


def baselineExtract(c_x, c_y):
    # detecting slope for line
    fit = np.polyfit(c_x, c_y, deg=1)

    # line points
    pt1 = (c_x[0], int(c_x[0] * fit[0] + fit[1]))
    pt2 = (c_x[-1], int(c_x[-1] * fit[0] + fit[1]))
    pt3 = (c_x[-1], int(c_x[0] * fit[0] + fit[1]))

    # fitted line
    cv2.line(image, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    # print("Baseline: ", fit[0])
    return fit[0]


def predictList(itemList):
    p = clf.predict([itemList])
    p1 = clf1.predict([itemList])
    p2 = clf2.predict([itemList])
    p3 = clf3.predict([itemList])

    print(p, p1, p2, p3)


clf = joblib.load("IE_svm.pkl")
clf1 = joblib.load("SN_svm.pkl")
clf2 = joblib.load("TF_svm.pkl")
clf3 = joblib.load("JP_svm.pkl")

# IMAGE load
image = cv2.imread("DATA/089_Shreyank.jpg")


# IMAGE Resize and cut boundaries
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
# sum_width = 0
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

        # Update features
        sum_height += h
        # sum_width += w

        c_x.append(int(x + (w / 2)))
        c_y.append(int(y + (h / 2)))

        # Getting ROI
        roi = gray[y:y + h, x:x + w]
        # show ROI
        # cv2.imshow('segment no:' + str(i), roi)
        # cv2.waitKey(0)

        sum_int_roi = 0
        int_count = 0
        for x2 in range(len(roi)):
            for i in range(len(roi[x2])):
                if roi[x2][i] < 220 and roi[x2][i] >= 90:
                    sum_int_roi += roi[x2][i]
                    int_count += 1

        avg_int_roi = sum_int_roi / int_count
        # print(avg_int_roi)
        sum_int += avg_int_roi
        # print(sum_int)
        # cv2.imshow("After changing ", roi)
        # cv2.waitKey(0)

        # t Creating ROI and Center on Image
        cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
        cv2.circle(image, (c_x[-1], c_y[-1]), 0, (255, 0, 0), 5)

        # Draw every contour and rectangle
        # cv2.drawContours(image, ctr, -1, (255,2,0), 3)
        # cv2.waitKey(0)

        if (count == 0):
            Rightmost = x + w
            # print("Rightmost", count, Rightmost)
            count += 1
            continue
        else:
            Leftmost = x
            # print("Leftmost", count, Leftmost)
            font = cv2.FONT_HERSHEY_PLAIN
            dist = Leftmost - Rightmost
            if dist > 0:
                cv2.line(image, (Leftmost, 1000), (Leftmost, 0), (0, 0, 0), thickness=1)
                cv2.putText(image, str(count) + "l", (Leftmost, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(image, (Rightmost, 1000), (Rightmost, 0), (0, 255, 0), thickness=1)
                cv2.putText(image, str(count) + "r", (Rightmost, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(dist), (Rightmost, 200), font, 1, (255, 255, 0), 1, cv2.LINE_AA)
                count += 1
                sum_dist += dist

        Rightmost = x + w

itemList.append(baselineExtract(c_x, c_y))
itemList.append(sum_height / count)
# itemList.append(sum_width / count)
itemList.append(sum_int / count)
itemList.append(sum_dist / count)
predictList(itemList)
