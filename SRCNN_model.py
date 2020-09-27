# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:48:46 2020

@author: fbasatemur
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam


# define the SRCNN model

class model:
      
      def srcnnModel(self):
          
          SRCNN = Sequential()
          
      
          SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                           activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
          
          SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                           activation='relu', padding='same', use_bias=True))
          
          SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                           activation='linear', padding='valid', use_bias=True))
          
          # define optimizer
          adam = Adam(lr=0.0003)
          
          # compile model
          SRCNN.compile(optimizer=adam, loss="mse", metrics="mse")
          
          return SRCNN
      
      
      def __init__(self):
            global srcnn
            srcnn = self.srcnnModel()
            srcnn.load_weights('3051crop_weight_200.h5')
            
            
      def get_model(self): 
            global srcnn
            return srcnn
      
      def __del__(self):
            global srcnn
            del srcnn

