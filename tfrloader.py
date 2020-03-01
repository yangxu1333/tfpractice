import os
import tensorflow as tf
import time
import numpy as np

default_timeit_steps = 10

def timeit(ds, steps=default_timeit_steps):
	start = time.time()
	it = iter(ds)
	for i in range(steps):
		next(it)
		if i%10 == 0:
			print('.',end='')
	#print()
	end = time.time()
	duration = end-start
	print("{} batches: {} s".format(steps, duration))
	print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

BATCH_SIZE = 20
TIME_STEP = 20
SHUFFLE_BUFFER_SIZE = 100
record_dir = r'C:\D\gits\UCF_VGG_TFR'
record_ds = tf.data.Dataset.list_files(record_dir+'\\*\\*')
record_ds = record_ds.shuffle(buffer_size = SHUFFLE_BUFFER_SIZE)
#tfr_ds = tf.data.TFRecordDataset(record_ds)
#parsed_ds = tfr_ds.map(_parse_feature_function)
CLASS_NAMES = np.array([item for item in os.listdir(record_dir)])
num_class = CLASS_NAMES.shape[0]

feature_format = {
    'dim': tf.io.FixedLenFeature([], tf.int64),
	'label': tf.io.FixedLenFeature([], tf.string),
    'features_raw': tf.io.FixedLenFeature([], tf.string),
}
def parse_tfr(tfr):
	feature_dict = tf.io.parse_example(tfr, feature_format)
	features = tf.io.parse_tensor(feature_dict['features_raw'], tf.float32)
	label = -1
	for i in range(num_class):
		if(CLASS_NAMES[i]==feature_dict['label']):
			label = i
			break
	return tf.reshape(features, [7*10*512]), label
def tf_parse_tfr(tfr):
	feature, label = tf.py_function(parse_tfr, [tfr], [tf.float32, tf.int32])
	feature.set_shape((7*10*512,))
	label.set_shape(())
	return feature, label
def process_path(path):
	tfr_ds = tf.data.TFRecordDataset(path)
	feature_label_ds = tfr_ds.map(tf_parse_tfr).batch(TIME_STEP, drop_remainder=True)
	return feature_label_ds
ds_start = process_path(next(iter(record_ds)))
start = True
for ds in record_ds:
	if(start):
		start = False
		continue
	ds_start = ds_start.concatenate(process_path(ds))
#labeled_ds = record_ds.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)
def prepare_ds(ds):
	#ds=ds.cache()
	ds=ds.batch(BATCH_SIZE)
	ds=ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
	return ds
train_ds = prepare_ds(ds_start)
#timeit(train_ds)
#print(str(next(iter(train_ds)))[:150])