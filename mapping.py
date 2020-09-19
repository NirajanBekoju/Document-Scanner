# import the necessary packages
import numpy as np
import cv2
def order_points(pts):
    pts = pts.sum(axis = 1)
    rect = np.zeros((4, 2), dtype = "float32")
   
    s = pts.sum(axis = 1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

   
    # return the ordered coordinates
    return rect