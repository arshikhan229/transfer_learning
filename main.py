# main.py
from data.data_loader import download_data, load_data
from scripts.train import train_model

def main():
    # Define paths and parameters
    train_dir = '/content/archive (6)/train'
    validation_dir = '/content/archive (6)/val'
    batch_size = 32
    img_size = (160, 160)
    num_classes = 2
    epochs = 10

    # Download and load data
    download_data()
    train_dataset, validation_dataset = load_data(train_dir, validation_dir, batch_size, img_size)

    # Train model
    model, history = train_model(train_dataset, validation_dataset, num_classes, epochs)

    # Save the model
    model.save('trained_model.h5')
    print("Model training complete and saved as 'trained_model.h5'")

if __name__ == "__main__":
    main()
