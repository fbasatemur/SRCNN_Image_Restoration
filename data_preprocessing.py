# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:48:10 2020

@author: fbasatemur
"""

import cv2
import numpy as np
import os
from PyQt5.QtGui import  QImage

#%% create low_resolution_images

def prepare_images(path, factor):
    
    # loop through the files in the directory
    for file in os.listdir(path):
        
        # open the file
        img = cv2.imread(path + '/' + file)
        
        # find old and new image dimensions
        height, width, _ = img.shape
        new_height = height / factor
        new_width = width / factor
        
        # resize the image - down
        img = cv2.resize(img, (int(new_width), int(new_height)), interpolation = cv2.INTER_LINEAR)
        
        # resize the image - up
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_LINEAR)
        
        # save the image
        print('Saving low_resolution_images/{}'.format(file))
        cv2.imwrite('low_resolution_images/{}'.format(file), img)
        
# prepare_images('images/', 2)

#%% model predict

def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    return image[border: -border, border: -border]


def predict(image_path, model):

    # load the degraded and reference images
    degraded = cv2.imread(image_path)
    
    # preprocess the image with modcrop
    degraded_mod = modcrop(degraded, 3)
    
    # convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(degraded_mod, cv2.COLOR_BGR2YCrCb)
    
    # create image slice and normalize  
    Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255.0
    
    # perform super-resolution with srcnn
    pre = model.predict(Y, batch_size=1)
    
    # post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    
    # copy Y channel back to image and convert to BGR
    temp = shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

    # return predict
    return output

def convert_cv_qt(cv_img):
      # Convert from an opencv image to QPixmap
      rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
      height, width, channels = rgb_image.shape
      bytes_per_line = channels * width
      convert_to_Qt_format = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
      return convert_to_Qt_format

      

