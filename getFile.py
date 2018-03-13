import cv2
import numpy as np


#import image
image = cv2.imread('./DATA/Baseline.jpg')
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
cv2.imshow('orig', image)
cv2.waitKey(0)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# threshold - binary thresholding to get clearer image.
ret, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('second', thresh)
cv2.waitKey(0)

# dilation - to get regions for segmentation
kernel = np.ones((3, 11), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated', img_dilation)
cv2.waitKey(0)

# kernel = np.ones((1, 3), np.uint8)
# img_dilation = cv2.erode(thresh, kernel, iterations=1)
# cv2.imshow('dilated', img_dilation)
# cv2.waitKey(0)
# thresh = 200
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# img_dilation = cv2.Canny(blur, thresh, thresh * 2)
# image = np.zeros(image.shape, np.uint8)  # Image to draw the contours

# find contours
im2, ctrs, hier = cv2.findContours(
    img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[
                     1])
word_inclination = 0
avg_area = 0
avg_h = 0
for ctr in sorted_ctrs:
    rect = cv2.minAreaRect(ctr)
    area = rect[1][0] * rect[1][1]
    avg_area += area
    avg_h += rect[1][1]

avg_area = avg_area / len(sorted_ctrs)
avg_h = avg_h / len(sorted_ctrs)

avg_skew = 0
count = 0
for i, ctr in enumerate(sorted_ctrs):

    # Get bounding box

    x, y, w, h = cv2.boundingRect(ctr)

    # if x < 10 or y < 10:
    #     continue
    # if w < 15 or h < 15:
    #     continue

    rect = cv2.minAreaRect(ctr)
    area = rect[1][0] * rect[1][1]

    # if rect[1][1] >= avg_h:  # area <= avg_area or
    #     continue

    count += 1
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    avg_skew += rect[2] * -1
    # print("----------------------------")
    # print(x)
    # print(y)
    # print(w)
    # print(h)
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(rect[2])
    # Getting ROI
    roi = image[y:y + h, x:x + w]
    # show ROI
    #cv2.imshow('segment no:' + str(i), roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (100, 0, 255), 1)
    # cv2.waitKey(0)
print("Count: " + str(count))
try:
    avg_skew = avg_skew / count
    avg_skew *= -1
    print("Average skew : " + str(avg_skew))
except Exception as e:
    print("Couldn't find Skew!")

cv2.imshow('marked areas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
