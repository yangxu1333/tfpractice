import os
import tensorflow as tf
import time
import numpy as np
feature_format = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'features_raw': tf.io.FixedLenFeature([], tf.string),
}
def _parse_feature_function(proto):
    return tf.io.parse_single_example(proto, feature_format)

default_timeit_steps = 10

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
    batch = next(it)
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

BATCH_SIZE = 20
record_dir = r'C:\D\gits\UCF_VGG_TFR'
record_ds = tf.data.Dataset.list_files(record_dir+'\\*\\*')
CLASS_NAMES = np.array([item for item in os.listdir(record_dir)])
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES
def get_features(record_file):
    tfr_ds = tf.data.TFRecordDataset(record_file)
    parsed_ds = tfr_ds.map(_parse_feature_function)
    batched_ds = parsed_ds.batch(BATCH_SIZE)
    return batched_ds
def process_path(path):
    return get_features(path), get_label(path)
labeled_ds = record_ds.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)
def prepare_ds(ds, shuffle_buffer_size = 1000):
    ds=ds.shuffle(buffer_size = shuffle_buffer_size)
    ds=ds.repeat()
    ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return ds
train_ds = prepare_ds(labeled_ds)
timeit(train_ds)