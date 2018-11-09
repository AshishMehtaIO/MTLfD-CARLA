# import tensorflow as tf
# import numpy as np
# import h5py
# import random
# import imgaug as ia
# from imgaug import augmenters as iaa
#
# f = h5py.File('./Final_dataset/Final_data.h5', 'r')
# shape = np.zeros([batch_size, f['targets'].shape[0]])
#
# train_filename = 'tensorflow_data.tfrecords'  # address to save the TFRecords file
# # open the TFRecords file
# writer = tf.python_io.TFRecordWriter(train_filename)
#
# for i in range(shape):
#     # print how many images are saved every 1000 images
#     if not i % 1000:
#         print
#         'Train data: {}/{}'.format(i, len(shape))
#         sys.stdout.flush()
#     # Load the image
#     img_front = f['Camera/RGB_front']
#     label = train_labels[i]
#     # Create a feature
#     feature = {'Camera/RGB_front': _int64_feature(label),
#                'Camera/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
#     # Create an example protocol buffer
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'Camera/RGB_front' : tf.train.Feature
#
#     }))
#
#     # Serialize to string and write on the file
#     writer.write(example.SerializeToString())
#
# writer.close()
# sys.stdout.flush()
#
# # Create the Example
# example = tf.train.Example(features=tf.train.Features(feature={
#     'Age': tf.train.Feature(
#         int64_list=tf.train.Int64List(value=[data['Age']])),
#     'Movie': tf.train.Feature(
#         bytes_list=tf.train.BytesList(
#             value=[m.encode('utf-8') for m in data['Movie']])),
#     'Movie Ratings': tf.train.Feature(
#         float_list=tf.train.FloatList(value=data['Movie Ratings'])),
#     'Suggestion': tf.train.Feature(
#         bytes_list=tf.train.BytesList(
#             value=[data['Suggestion'].encode('utf-8')])),
#     'Suggestion Purchased': tf.train.Feature(
#         float_list=tf.train.FloatList(
#             value=[data['Suggestion Purchased']])),
#     'Purchase Price': tf.train.Feature(
#         float_list=tf.train.FloatList(value=[data['Purchase Price']]))
# }))


from pygit2 import Repository

# import logging
# import sys
#
# file_handler = logging.FileHandler(filename='tmp.log')
# stdout_handler = logging.StreamHandler(sys.stdout)
# handlers = [file_handler, stdout_handler]
#
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
#     handlers=handlers
# )
#
# logger = logging.getLogger('LOGGER_NAME')

print(Repository('.').head.shorthand)