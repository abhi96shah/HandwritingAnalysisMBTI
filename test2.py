import cv2
import numpy as np

#<------FINDING CONTOURS------>
def findContours(image, dst):

	#dilation
	kernel = np.ones((1,12), np.uint8)
	img_dilation = cv2.dilate(dst, kernel, iterations=1)
	
	rows,cols = image.shape[:2]

	#find contours
	im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#sort contours
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0], reverse = False)

	return sorted_ctrs

	
#<------HOUGH TRANSFORM------>
def houghBase(image, edges):

	kernel = np.ones((3,100), np.uint8)
	edges_dilated = cv2.dilate(edges, kernel, iterations=1)
	cv2.imshow('dilated edges',edges_dilated)
	cv2.waitKey(0)


	lines = cv2.HoughLinesP(image=edges_dilated,rho=0.5,theta=np.pi/180, threshold=300, lines=np.array([]), minLineLength=500, maxLineGap=10)

	a,b,c = lines.shape
	for i in range(a):
	    print(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3])
	    cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)

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

	print(fit[0])
#<------END BASELINE FIT------>


#<------READ IMAGE------>
image = cv2.imread('trial.jpg')
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#<------PREPROCESS IMAGE------>
#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('second',thresh)
cv2.waitKey(0)

# What to do? 
#de-noise
dst = cv2.fastNlMeansDenoising(thresh,None,10,21,7)
cv2.imshow('de-noise',dst)
cv2.waitKey(0)

# # #eroded
# kernel_erode = np.ones((45,0), np.uint8)
# img_erode= cv2.erode(img_dilation, kernel_erode, iterations=1)
# cv2.imshow('eroded',img_erode)
# cv2.waitKey(0)

# #edges
# edges = cv2.Canny(dst,100,200,apertureSize = 5)
# cv2.imshow('edges',edges)
# cv2.waitKey(0)

# kernel = np.ones((1,20), np.uint8)
# edges_dilated = cv2.dilate(edges, kernel, iterations=1)
# cv2.imshow('dilated edges',edges_dilated)
# cv2.waitKey(0)

#<------!PREPROCESS IMAGE------>

#houghBase(image, edges)

sorted_ctrs = findContours(image, dst)

#<------EXTRACTING FEATURES------>
sum_height = 0
sum_width = 0
count = 0
c_x = []
c_y = []
Leftmost = 0
Rightmost = 0

for i, ctr in enumerate(sorted_ctrs):
	if (cv2.contourArea(ctr) > 250.0):
		# Get bounding box
		x, y, w, h = cv2.boundingRect(ctr)
		
		#Update features
		sum_height+=h
		sum_width+=w
		count+=1
		c_x.append(int(x + (w/2)))
		c_y.append(int(y + (h/2)))
		
		# Getting ROI
		roi = image[y:y+h, x:x+w]
		# show ROI
		#cv2.imshow('segment no:'+str(i),roi) 
		#cv2.waitKey(0)
		
		# Creating ROI and Center on Image
		# cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
		# cv2.circle(image, (c_x[-1], c_y[-1]), 0, (255, 0, 0), 5)

		# Draw every contour and rectangle
		# cv2.drawContours(image, ctr, -1, (255,2,0), 3)
		# cv2.waitKey(0)

		
		if (i==0):
			Rightmost = tuple(ctr[ctr[:,:,0].argmax()][0])
			continue
		else:
			Leftmost = tuple(ctr[ctr[:,:,0].argmin()][0])
			dist = np.sqrt(((Leftmost[0]-Rightmost[0])**2) + ((Leftmost[1] - Rightmost[1])**2))
			print(dist)

		Rightmost = tuple(ctr[ctr[:,:,0].argmax()][0])
		print(Rightmost)




print(count)
print(sum_height/count)
print(sum_width/count)
# avg_width = sum_width/count

baselineExtract(c_x, c_y)


cv2.imshow('marked areas',image)
cv2.waitKey(0)

