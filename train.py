# scripts/train.py
import tensorflow as tf
from models.model_setup import setup_model, compile_model

def train_model(train_dataset, validation_dataset, num_classes, epochs=10):
    model = setup_model(num_classes)
    model = compile_model(model)

    history = model.fit(train_dataset,
                        validation_data=validation_dataset,
                        epochs=epochs)
    return model, history
