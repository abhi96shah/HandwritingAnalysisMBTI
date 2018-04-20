
import cv2
import numpy as np
import statistics
import csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.externals import joblib
import matplotlib.pyplot as plt

baselineList = list()
#intensityList = list()
distWordsList = list()
# avgHeightList = list()
# avgWidthList = list()
featureList = list()

#<------FINDING CONTOURS------>


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


classList_IE = list()
classList_SN = list()
classList_TF = list()
classList_JP = list()

#<------READ IMAGE------>

count_image = 1
with open('Dataset.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if(row['Sample'] == "y" and row['MBTI']):
            itemList = list()

            MBTI_whole = row['MBTI']
            MBTI_IE = MBTI_whole[0]
            MBTI_SN = MBTI_whole[1]
            MBTI_TF = MBTI_whole[2]
            MBTI_JP = MBTI_whole[3]

            classList_IE.append(MBTI_IE)
            classList_SN.append(MBTI_SN)
            classList_TF.append(MBTI_TF)
            classList_JP.append(MBTI_JP)

            print(row['Name'])
            # print(classList)
            if(count_image < 10):
                name = "./DATA/00" + str(count_image) + "_" + str(row['Name']) + ".jpg"
            elif(count_image > 9 and count_image < 100):
                name = "./DATA/0" + str(count_image) + "_" + str(row['Name']) + ".jpg"
            image = cv2.imread(name)

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
                    # print("Rightmost", count, Rightmost)

            # print("Number of words detected :", count)
            # print("Average Height: ", sum_height / count)
            # print("Average Width: ", sum_width / count)
            # print("Average Intensity: ", sum_int / count)
            # print("Average Distance between words: ", sum_dist / count - 1)
            baselineList.append(baselineExtract(c_x, c_y))
            # avgHeightList.append(sum_height / count)
            # avgWidthList.append(sum_width / count)
            # intensityList.append(sum_int / count)
            distWordsList.append(sum_dist / count)
            itemList.append(baselineExtract(c_x, c_y))
            itemList.append(sum_height / count)
            # itemList.append(sum_width / count)
            itemList.append(sum_int / count)
            itemList.append(sum_dist / count)
            # print(itemList)
            featureList.append(itemList)
            # print(featureList)
            # cv2.imshow('marked areas', image)
            # cv2.waitKey(0)
        count_image += 1


X_train, X_test, y_train, y_test = train_test_split(featureList, classList_IE, random_state=0)

svm_model_rbf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
svm_predictions = svm_model_rbf.predict(X_test)

accuracy = svm_model_rbf.score(X_test, y_test)
print(accuracy)

X1_train, X1_test, y1_train, y1_test = train_test_split(featureList, classList_SN, random_state=0)

svm_model_rbf1 = SVC(kernel='rbf', C=1).fit(X1_train, y1_train)
svm_predictions1 = svm_model_rbf1.predict(X1_test)

accuracy1 = svm_model_rbf1.score(X1_test, y1_test)
print(accuracy1)

X2_train, X2_test, y2_train, y2_test = train_test_split(featureList, classList_TF, random_state=0)

svm_model_rbf2 = SVC(kernel='rbf', C=1).fit(X2_train, y2_train)
svm_predictions2 = svm_model_rbf2.predict(X2_test)

accuracy2 = svm_model_rbf2.score(X2_test, y2_test)
print(accuracy2)

X3_train, X3_test, y3_train, y3_test = train_test_split(featureList, classList_JP, random_state=0)

svm_model_rbf3 = SVC(kernel='rbf', C=1).fit(X3_train, y3_train)
svm_predictions3 = svm_model_rbf3.predict(X3_test)

accuracy3 = svm_model_rbf3.score(X3_test, y3_test)
print(accuracy3)
print(X3_test)


joblib.dump(svm_model_rbf, "IE_svm.pkl")
joblib.dump(svm_model_rbf1, "SN_svm.pkl")
joblib.dump(svm_model_rbf2, "TF_svm.pkl")
joblib.dump(svm_model_rbf3, "JP_svm.pkl")
