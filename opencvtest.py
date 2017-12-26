import cv2

image = cv2.imread('test_images/solidWhiteCurve.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray_image)
cv2.waitKey(0)