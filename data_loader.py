# data/data_loader.py
import tensorflow as tf
import os

def download_data():
    from google.colab import files
    files.upload()

    !kaggle datasets download -d vuppalaadithyasairam/bone-fracture-detection-using-xrays
    !unzip "/content/bone-fracture-detection-using-xrays.zip"

def load_data(train_dir, validation_dir, batch_size=32, img_size=(160, 160)):
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                image_size=img_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                     shuffle=True,
                                                                     batch_size=batch_size,
                                                                     image_size=img_size)

    return train_dataset, validation_dataset
