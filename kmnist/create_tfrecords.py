"""
Module to create tfrecords that contains kmist examples.
"""
import os
import numpy as np
import tensorflow as tf

from utils import serialize_example


DATA_DIR = "/data"
SEED = 1227

def main():
    with np.load(os.path.join(DATA_DIR, "kmnist-train-imgs.npz")) as img_container, \
        np.load(os.path.join(DATA_DIR, "kmnist-train-labels.npz")) as label_container:

        images = img_container["arr_0"]
        labels = label_container["arr_0"]

    assert images.shape[0] == labels.shape[0] # All images have labels

    kmnist_ds = tf.data.Dataset.from_tensor_slices((images, labels))

    kmnist_ds = kmnist_ds.shuffle(70000, seed=SEED)

    kmnist_train_ds = kmnist_ds.skip(1000)
    kmnist_dev_ds = kmnist_ds.take(1000)

    with tf.io.TFRecordWriter("/data/kmnist_dev.tfrecord") as writer:
        for image, label in kmnist_dev_ds:
            example = serialize_example(image, label)
            writer.write(example)

    with tf.io.TFRecordWriter("/data/kmnist_train.tfrecord") as writer:
        for image, label in kmnist_train_ds:
            example = serialize_example(image, label)
            writer.write(example)

if __name__ == "__main__":
    main()