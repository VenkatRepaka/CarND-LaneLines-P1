# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 09:17:24 2017

@author: repvenk
"""
import cv2

vidcap = cv2.VideoCapture('challenge_frames/challenge.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print ('Read a new frame: ', success)
  cv2.imwrite("challenge_frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1