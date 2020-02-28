import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16

# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

dbg_file_limit = 5
path = r'C:\D\gits\UCF101'
record_dir = r'C:\D\gits\UCF_VGG_TFR'
vgg_model = VGG16(include_top=False, weights='imagenet',input_shape=(240,320,3))
#vgg_model.summary()
for foldername in os.listdir(path):
	if(not os.access(record_dir+'\\'+foldername,os.F_OK)):
		os.mkdir(record_dir+'\\'+foldername)
	count = 0
	for filename in os.listdir(path+'\\'+foldername):
		count+=1
		if(dbg_file_limit != 0 and count > dbg_file_limit):
			break
		video = cv2.VideoCapture(path+'\\'+foldername+'\\'+filename)
		record_file = record_dir+'\\'+foldername+'\\'+filename+'.tfrecords'
		if(os.access(record_file,os.F_OK)):
			os.remove(record_file)
		with tf.io.TFRecordWriter(record_file) as writer:
			while(1):
				ret_val,frame = video.read()
				if not(ret_val):
					break
				#print(frame.shape)
				frame_expanded = np.expand_dims(frame, axis=0)#(1,240,320,3)
				#print(frame_expanded.shape)
				features = vgg_model.predict(frame_expanded)
				#print(features.shape)#(1,7,10,512)
				feature_format = {
					'height': _int64_feature(features.shape[0]),
					'width': _int64_feature(features.shape[1]),
					'depth': _int64_feature(features.shape[2]),
					'features_raw': _bytes_feature(tf.io.serialize_tensor(features)),
				}
				x=tf.train.Example(features=tf.train.Features(feature=feature_format))
				writer.write(x.SerializeToString())