import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2


def resize_image(img):
    resized_image = cv2.resize(img, (48, 48)).astype('float32')
    resized_image = np.expand_dims(resized_image, -1)
    resized_image /= 127.5
    resized_image -= 1.
    reshaped_image = resized_image.reshape(1, 48, 48, 1)
    return reshaped_image


def loadModel(path):
    model=load_model(path)
    return model
