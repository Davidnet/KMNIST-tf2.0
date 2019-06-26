"""
Module that contains utilites for the kmnist dataset.
"""

import numpy as np
import tensorflow as tf
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    serlialized_image = tf.io.serialize_tensor(image).numpy()

    feature = {
        'image_raw': _bytes_feature(serlialized_image),
        'label': _int64_feature(label.numpy())
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _parse_function(example_proto):
  # Parse the input tf.Example proto using the dictionary below.
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def parse_features(kmnist_ds):

    def _transform_img(img_dict):

        image = tf.io.parse_tensor(img_dict["image_raw"], tf.uint8)

        return dict(
            image = tf.reshape(image, [28, 28]) ,
            label = img_dict["label"]
        )


    kmnist_ds = kmnist_ds.map(_parse_function)


    return kmnist_ds.map(_transform_img)