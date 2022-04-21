import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def line_detection(img):
    #edge_height, edge_width = img.shape[:2]

    rho_range = np.round(np.sqrt(np.square(img.shape[0]) + np.square(img.shape[1])))

    thetas = np.arange(0, 180, 1)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    #print(rho_range)
    accumulator = np.zeros((2 * int(rho_range), len(thetas)), dtype=np.uint8)
    edge_pixels = np.where(img == 255)
    coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

    for i in range(len(coordinates)):
        for j in range(len(thetas)):
            rho_idx = int(round(coordinates[i][1] * cos_thetas[j]
                      + coordinates[i][0] * sin_thetas[j]))
            accumulator[rho_idx, j] +=2
    return accumulator




img = cv.imread('images/road_lane.jpg')
#cv.imshow('road_lane', img)

resized_img = cv.resize(img, (600,600))

cv.imshow('resized_img', resized_img)

# converting to grayscale
gray_img = cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)
#cv.imshow('Gray_img', gray_img)

blur_img = cv.GaussianBlur(gray_img, (3, 3), 1)
#cv.imshow("blur", blur_img)
# Edge detection using canny
canny_img = cv.Canny(blur_img, 50,200)
#cv.imshow('Canny', canny_img)

# dilating the image
dilated = cv.dilate(canny_img, (7,7), iterations=1)
#cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (3,3), iterations=1)
#cv.imshow('eroded', eroded)

accumulator = line_detection(eroded)
threshold_value = 245
edge_pixels = np.where(accumulator > threshold_value)
coordinates = list(zip(edge_pixels[0], edge_pixels[1]))

for i in range(0, len(coordinates)):
    a = np.cos(np.deg2rad(coordinates[i][1]))
    b = np.sin(np.deg2rad(coordinates[i][1]))
    x0 = a*coordinates[i][0]
    y0 = b*coordinates[i][0]

    #this will help us to get the extreme points to form a line
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv.line(resized_img, (x1,y1), (x2,y2), (255,0,0), 2)

cv.imshow("final img", resized_img)
plt.imshow(accumulator)
plt.show()
cv.waitKey(0)
