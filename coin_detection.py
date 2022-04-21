from functools import total_ordering
from tabnanny import check
import cv2 as cv
import numpy as np
import random
import math

def detect_circle(img):
    height, width = img.shape[:2]
    x_range = np.arange(0, width, step=1)
    y_range = np.arange(0, height, step=1)
    radius_range = np.arange(0,np.sqrt(np.square(width) + np.square(height)), step = 1)
    accumulator = np.zeros((len(radius_range), len(x_range), len(y_range)))

    nonzero_y_values, nonzero_x_values = np.nonzero(img)

    for i in range(len(nonzero_y_values)):
        for x in x_range:
            for y in y_range:
                dist = np.sqrt(np.square(nonzero_x_values[i] - x) + np.square(nonzero_y_values[i] - y))
                r = round(dist)
                accumulator[r][x][y] +=1
    
    threshold_value = 40
    circles = np.argwhere(accumulator > threshold_value)
    return circles
    


img = cv.imread('images/c.jpeg')
print(img.shape)

img = cv.resize(img, (100,100))
cv.imshow('coins', img)
gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('Gray_img', gray_img)

blur_img = cv.GaussianBlur(gray_img, (3, 3), 1)
#cv.imshow('Blur', blur_img)
canny_img = cv.Canny(blur_img, 200,200)
#cv.imshow('coins', canny_img)
list_of_circles = detect_circle(canny_img)

for c in list_of_circles:
        radius,x_val,y_val = c
        cv.circle(img, (x_val,y_val), radius,(0,255,0), 1)
        
#img = cv.resize(img, (400,400))
cv.imshow('final', img)

cv.waitKey(0)
 
