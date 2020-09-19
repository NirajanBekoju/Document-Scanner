import numpy as np
import cv2

# Reading the images
original_image = cv2.imread('images/1.jpg')
# Image Resizing
width = original_image.shape[1]
height = original_image.shape[0]
src = cv2.resize(original_image, (int(width/5), int(height/5)))

# Initialing the trackbar
def nothing(x):
    pass

cv2.namedWindow('Controller')
cv2.createTrackbar('Threshold', 'Controller', 127, 255, nothing)

# Image Conversion to Grayscale
img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

while True:
    # Image Thresholding
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    threshold_value = cv2.getTrackbarPos('Threshold', 'Controller')
    ret, thresh = cv2.threshold(img_blur, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Coutour Detection
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        biggest_contour = contours[0]
        biggest_contour_area = cv2.contourArea(biggest_contour)

        for contour in contours:
            # Check if the contout is rectangle or not
            contour_area = cv2.contourArea(contour) 
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            # If the contour is rectangle 
            if len(approx) == 4:
                if contour_area > biggest_contour_area:
                    biggest_contour = contour
                    biggest_contour_area = contour_area
            
        contour_image = cv2.drawContours(src, biggest_contour, -1, (0,255,0), 3)
        # Image Viewer
        # cv2.imshow("Original Image", src)
        # cv2.imshow("Grayscale Image", img_gray)
        cv2.imshow("Threshold Image",thresh)
        cv2.imshow("Contour Image",contour_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print(biggest_contour_area)
print(biggest_contour)

# Closing Tags
cv2.destroyAllWindows()