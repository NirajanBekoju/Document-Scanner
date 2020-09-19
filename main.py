import cv2
import numpy as np
import mapping

# Declaring the variables
black_image = np.zeros((500, 600, 3), np.uint8)
kernel = np.ones((5,5), np.uint8)
document_point = np.float32([[0,0], [500, 0], [500, 600], [0, 600]])

# Reading the images
original_image = cv2.imread('images/7.jpg')

# Image Resizing
width = original_image.shape[1]
height = original_image.shape[0]
src = cv2.resize(original_image, (int(width/5), int(height/5)))
org = src.copy()

# Image Conversion to Grayscale
img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# Thresholding the image
img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
ret, thresh = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY)

# Dilation of the image
img_gray_dilate = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow("Dilated Image", img_gray_dilate)

# Canny Edge Detection
edged_image = cv2.Canny(img_gray_dilate, 30, 50)
cv2.imshow("Canny", edged_image)

# Finding the contours
contours, hierarchy = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Check if the biggest contour is rectangle or not
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04*perimeter, True)
    if len(approx) == 4:
        target = approx
        break

cv2.drawContours(src, contours, -1, (255,0,0), 3)
cv2.imshow("Contour Image", src)

# Vertices Point Approximation
approx = mapping.order_points(target)

op = cv2.getPerspectiveTransform(approx, document_point)
dst = cv2.warpPerspective(org, op, (500, 600))

cv2.imshow("Scanned Image", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()