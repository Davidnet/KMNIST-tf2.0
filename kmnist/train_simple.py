"""
Module that contain the main logic for train for a simple MLP.
"""
import tensorflow as tf

from utils import parse_features


train_ds = tf.data.TFRecordDataset(["/data/kmnist_train.tfrecord"])
dev_ds = tf.data.TFRecordDataset(["/data/kmnist_dev.tfrecord"])

train_ds = parse_features(train_ds)
dev_ds = parse_features(dev_ds)

inputs = tf.keras.Input(shape=(784,), name="Hiraganas")
x = tf.keras.layers.Dense(64, activation="relu")(inputs)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

adam_optimizer = tf.keras.optimizers.Adam()

loss_categorical = tf.keras.losses.SparseCategoricalCrossentropy()

metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]



model.compile(optimizer=adam_optimizer,
              loss=loss_categorical,
              metrics=metrics)

### Simple model

def reshape_and_normalize(record_dict):
    image = tf.cast(tf.reshape(record_dict["image"], [784]), tf.float64)
    image = tf.math.divide(image, 255.0)
    label = record_dict["label"]
    return (image, label)

train_ds_simple = train_ds.map(reshape_and_normalize)

dev_ds_simple = dev_ds.map(reshape_and_normalize)

train_ds_simple = train_ds_simple.shuffle(buffer_size=70000).batch(32)
dev_ds_simple = dev_ds_simple.batch(64)

history = model.fit(train_ds_simple, epochs=5, validation_data=dev_ds_simple)


print(f"history\n : {history.history}")