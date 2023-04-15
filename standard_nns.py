import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input
import cv2
import os
import numpy as np
import pandas as pd
from model_testing import preproc_data 


def create_cnn_model():
    model = Sequential()
    model.add(Dense(128, activation='relu',input_shape=(256,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(1, activation='linear'))  # Output layer with linear activation for regression
    return model
def run_test(data_path,verbose = False):
    def log(x):
        if verbose:
            print(x)
    
    df_reg = preproc_data(data_path)
df_reg = run_test()

X_train,X_test,y_train,y_test = train_test_split(df_reg.drop("price",axis=1),df_reg["price"],test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled.shape)