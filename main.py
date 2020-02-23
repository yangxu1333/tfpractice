import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16

path = r'C:\D\gits\UCF101'
dirs = os.listdir(path)
print(dirs[0])
files = os.listdir(path+'\\'+dirs[0])
print(files[0])
video = cv2.VideoCapture(path+'\\'+dirs[0]+'\\'+files[0])
# while(1):
ret_val,frame = video.read()
""" if not(ret_val):
    break """
print(frame.shape)
frame_expanded = np.expand_dims(frame, axis=0)#(1,240,320,3)
print(frame_expanded.shape)
vgg_model = VGG16(include_top=False, weights='imagenet',input_shape=(240,320,3))
#vgg_model.summary()
features = vgg_model.predict(frame_expanded)
print(features.shape)#(1,7,10,512)