import tensorflow as tf
# print(tf.test.gpu_device_name())

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import cv2
import os
import numpy as np
import pandas as pd
# Define the path to your directory containing .png images
image_dir = "data//landsat//"



def extract_label_from_filename(f):
    df = pd.read_csv("data//funda_sample_ams_geocoded_gs.csv",index_col=0)

    price = df[df["house_id"]==int(f.split(".")[0])]["price"]

    return price
# Initialize empty lists to store the images and their corresponding labels
X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []
# Loop through the .png images in the directory
for filename in os.listdir(image_dir)[:int(0.6*len(os.listdir(image_dir)))]:
    if filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)  # Use cv2.imread() from OpenCV for loading images
        # Resize image to 256x256
        image = cv2.resize(image, (256, 256))
        # Preprocess the image as needed (e.g., resize, normalize, augment)
        # Add the preprocessed image to X_train
        X_train.append(image)
        # Extract the label from the filename or any other source (e.g., metadata)
        label = extract_label_from_filename(filename)
        # Add the label to y_train
        y_train.append(label)

for filename in os.listdir(image_dir)[int(0.6*len(os.listdir(image_dir))):int(0.8*len(os.listdir(image_dir)))]:
    if filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)  # Use cv2.imread() from OpenCV for loading images
        image = cv2.resize(image, (256, 256))
        # Preprocess the image as needed (e.g., resize, normalize, augment)
        # Add the preprocessed image to X_train
        X_val.append(image)
        # Extract the label from the filename or any other source (e.g., metadata)
        label = extract_label_from_filename(filename)
        # Add the label to y_train
        y_val.append(label)
        
for filename in os.listdir(image_dir)[int(0.8*len(os.listdir(image_dir))):]:
    if filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)  # Use cv2.imread() from OpenCV for loading images
        # Preprocess the image as needed (e.g., resize, normalize, augment)
        image = cv2.resize(image, (256, 256))
        # Add the preprocessed image to X_train
        X_test.append(image)
        # Extract the label from the filename or any other source (e.g., metadata)
        label = extract_label_from_filename(filename)
        # Add the label to y_train
        y_test.append(label)
# Convert the lists to NumPy arrays for further processing
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Define the CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))  # Output layer with linear activation for regression
    return model

# Create an instance of the CNN model
model = create_cnn_model()

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Use mean squared error (mse) as the loss function

# Train the model
# Assuming you have your training data (X_train, y_train) and validation data (X_val, y_val) ready
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val))

# Evaluate the model
# Assuming you have your test data (X_test, y_test) ready
loss = model.evaluate(X_test, y_test)
print("Test loss: ", loss)

# Make predictions
# Assuming you have your input data (X_pred) ready for making predictions
# predictions = model.predict(X_pred)