# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:50:48 2017

@author: repvenk
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque

#show_images = True
show_images = False

last_20_Slope_Const = deque(maxlen=20)

def show_image(image):
    if(show_images == True):
        plt.imshow(image)
        plt.show()
        plt.pause(0.5)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def slope(x1, x2, y1, y2):
    return (y2-y1)/(x2-x1)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    img_shape = img.shape
#     lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    left_line_coordinates = [[], []]
    right_line_coordinates = [[], []]
    min_y = 0
    if(img_shape[1] > 1200) :
        min_y = 450
    else:
        min_y = 325
#    if(lines is None):
#        print('I am none')
#    print(lines)
    if(lines is not None):
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope_of_line = slope(x1, x2, y1, y2)
    #            print(line, slope_of_line)
    #            if(slope_of_line < -0.75 and slope_of_line > -3):
                if(slope_of_line < -0.5):
                    left_line_coordinates[0].append(x1)
                    left_line_coordinates[0].append(x2)
                    left_line_coordinates[1].append(y1)
                    left_line_coordinates[1].append(y2)
    #            elif(slope_of_line > 0.5 and slope_of_line < 3):
                elif(slope_of_line > .5):
                    right_line_coordinates[0].append(x1)
                    right_line_coordinates[0].append(x2)
                    right_line_coordinates[1].append(y1)
                    right_line_coordinates[1].append(y2)
    #    print(left_line_coordinates)
    #    print(right_line_coordinates)
        if(len(left_line_coordinates[0]) == 0 or len(left_line_coordinates[1]) == 0 \
           or len(right_line_coordinates[0]) == 0 or len(right_line_coordinates[1]) == 0) :
            return img
        else:
            left_slope_constant = np.polyfit(left_line_coordinates[0], left_line_coordinates[1], 1)
            right_slope_constant = np.polyfit(right_line_coordinates[0], right_line_coordinates[1], 1)
            
            slope_left = left_slope_constant[0]
            const_left = left_slope_constant[1]
            slope_right = right_slope_constant[0]
            const_right = right_slope_constant[1]
            max_x_left = max(left_line_coordinates[0])
            min_y_left =img_shape[0]
            min_x_left = int((min_y_left - const_left)/slope_left)
            max_y_left = int(max_x_left*slope_left + const_left)
            
            max_x_left = int((min_y - const_left)/slope_left)
            max_y_left = min_y
            
            min_x_right = min(right_line_coordinates[0])
            min_y_right = int(min_x_right*slope_right + const_right)
            max_y_right = img_shape[0]
            max_x_right = int((max_y_right - const_right)/slope_right)
            
            min_x_right = int((min_y - const_right)/slope_right)
            min_y_right = min_y
            
            line_left = [min_x_left, min_y_left, max_x_left, max_y_left]
            line_right = [min_x_right, min_y_right, max_x_right, max_y_right]
            lines = [line_left, line_right]
            cv2.line(img, (min_x_left, min_y_left), (max_x_left, max_y_left), [255, 0, 0], thickness=7)
            cv2.line(img, (min_x_right, min_y_right), (max_x_right, max_y_right), [255, 0, 0], thickness=7)
    else:
        left_slope_constant, right_slope_constant = np.mean(last_20_Slope_Const, axis=0)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def processImage(image):
    copied_image = np.copy(image)
    gray_scale_image = grayscale(copied_image)
    show_image(gray_scale_image)
    gaussed_image = gaussian_blur(gray_scale_image, 5)
    show_image(gaussed_image)
    canny_edges_image = canny(gaussed_image, 80, 240)
    show_image(canny_edges_image)
    shape = image.shape
    if(shape[1] > 1200):
        vertices = np.array([[(180,shape[0]), (600, 450), (760, 450), (1200, shape[0])]], dtype=np.int32)
    else:
        vertices = np.array([[(160,image.shape[0]), (410, 340), (550, 340), (900, image.shape[0])]], dtype=np.int32)
    masked_image = region_of_interest(canny_edges_image, vertices)
    show_image(masked_image)
    
    rho = 2
    theta = np.pi/180
    threshold = 15
    minLineLength = 40
    maxLineGap = 20
    hough_line_image = hough_lines(masked_image, rho, theta, threshold, minLineLength, maxLineGap)
    show_image(hough_line_image)
    weighted_image = weighted_img(hough_line_image, image)
    return weighted_image

def normalizeImage(image):
    # Added code from blog http://akash0x53.github.io/blog/2013/04/29/RGB-Normalization/
    normalized_image = np.zeros(image.shape)
    b=image[:,:,0]
    g=image[:,:,1]
    r=image[:,:,2]
    sum = b+g+r
    normalized_image[:,:,0]=b/sum*255.0
    normalized_image[:,:,1]=g/sum*255.0
    normalized_image[:,:,2]=r/sum*255.0
    return cv2.convertScaleAbs(normalized_image)
    

#test_image = mpimg.imread("challenge_frames/frame81.jpg")
#test_image = mpimg.imread("challenge_frames/frame109.jpg")
#test_image = mpimg.imread("challenge_frames/frame137.jpg")
#test_image = mpimg.imread("test_images/solidWhiteCurve.jpg")
#test_image = mpimg.imread("test_images/solidYellowLeft.jpg")
#show_image(test_image)
#
#plt.imshow(processImage(test_image))
#plt.show()
#plt.pause(0.5)
    
#test_images = os.listdir("test_images/")
#for index, test_image_str in enumerate(test_images):
#   test_image_abs_path = ''. join(["test_images/", test_image_str])
#   test_image_output_abs_path = ''. join(["test_images_output/", test_image_str])
#   test_image = mpimg.imread(test_image_abs_path)
#   processed_image = processImage(test_image)
#   cv2.imwrite(test_image_output_abs_path, processed_image)

#clip_solid_white_output = 'test_videos_output/solidWhiteRight.mp4'
#clip_solid_white = VideoFileClip("test_videos/solidWhiteRight.mp4")
#challenge_clip = clip_solid_white.fl_image(processImage) #NOTE: this function expects color images!!
#challenge_clip.write_videofile(clip_solid_white_output, audio=False)

clip_solid_yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip_solid_yellow = VideoFileClip("test_videos/solidYellowLeft.mp4")
challenge_clip = clip_solid_yellow.fl_image(processImage) #NOTE: this function expects color images!!
challenge_clip.write_videofile(clip_solid_yellow_output, audio=False)
#
#clip_challenge_output = 'test_videos_output/challenge.mp4'
#clip_challenge = VideoFileClip("test_videos/challenge.mp4")
#challenge_clip = clip_challenge.fl_image(processImage) #NOTE: this function expects color images!!
#challenge_clip.write_videofile(clip_challenge_output, audio=False)