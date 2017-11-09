# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 06:08:20 2017

@author: repvenk
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image = mpimg.imread('challenge_frames/frame108.jpg')
plt.imshow(image)
plt.show()
plt.pause(0.5)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image)
plt.show()
plt.pause(0.5)

ret, mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(image,image,mask = mask_inv)
plt.imshow(img1_bg)
plt.show()
plt.pause(0.5)


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
plt.imshow(hsv_image)
plt.show()
plt.pause(0.5)



boundaries = [
	([150, 0, 150], [255, 0, 255])
]

for (lower, upper) in boundaries:
# create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    
    plt.imshow(output)
    plt.show()
    plt.pause(0.5)
       
#    cv2.imshow("images", np.hstack([image, output]))
#    cv2.waitKey(0)