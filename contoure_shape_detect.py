import cv2
import numpy as np

image = cv2.imread('images/shapes.png')

# step 1: convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# step 2: apply canny edge detector
lower_threshold = 100
upper_threshold = 150
canny_image = cv2.Canny(gray_image, lower_threshold, upper_threshold)

# step 3: find the contours
contours, hierarchy = cv2.findContours(canny_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# step 4: find length of the contours
print("Length of the contours: ", len(contours))

# step 5: draw the contours
copy_main_image = image.copy()
cv2.drawContours(copy_main_image, contours, -1, (0, 255, 0), 3)

cv2.imshow('image', image)
cv2.imshow('gray_image', gray_image)
cv2.imshow('canny_image', canny_image)
cv2.imshow('contours', copy_main_image)
cv2.waitKey(0)
