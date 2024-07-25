# models/model_setup.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def setup_model(num_classes):
    base_model = MobileNetV2(input_shape=(160, 160, 3),
                             include_top=False,
                             weights='imagenet')

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    return model

def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
