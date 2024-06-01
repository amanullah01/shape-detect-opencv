import cv2
import numpy as np

image = cv2.imread('images/shapes.png')
copy_main_image = image.copy()

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

for cnt in contours:
    # find the area each of the contour
    area = cv2.contourArea(cnt)
    print("Area of the contour: ", area)

    # step 5: draw the contours

    cv2.drawContours(copy_main_image, cnt, -1, (0, 255, 0), 3)

    # step 6: find the arc length of contours
    perimeter = cv2.arcLength(cnt, True)

    # Step 7: find the corner points each of the shape
    approxCornerPoints = cv2.approxPolyDP(cnt, perimeter*0.02, True)

    # step 8: Length of the corner points each of the shape
    print("corner point length: ", len(approxCornerPoints))
    objectCorner = len(approxCornerPoints)

    # Step 9: draw bounding box
    x, y, w, h = cv2.boundingRect(approxCornerPoints)
    cv2.rectangle(copy_main_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    if objectCorner == 3:
        object_type = 'Triangle'
    elif objectCorner == 4:
        aspect_ratio = w/float(h)
        if 0.98 < aspect_ratio < 1.03:
            object_type = 'Square'
        else:
            object_type = 'Rectangle'
    elif objectCorner > 4:
        object_type = 'Circle'
    else:
        object_type = "None"

    cv2.putText(copy_main_image, object_type, (x+(w//2)-20, y+(h//2)-1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


cv2.imshow('image', image)
# cv2.imshow('gray_image', gray_image)
# cv2.imshow('canny_image', canny_image)
cv2.imshow('Final Image', copy_main_image)
cv2.waitKey(0)
