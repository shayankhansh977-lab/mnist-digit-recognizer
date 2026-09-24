import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import mnist
from PIL import Image

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images_reshaped = train_images.reshape((60000, 28*28)).astype('float') / 255
test_images_reshaped = test_images.reshape((10000, 28*28)).astype('float') / 255

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(28*28,)),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images_reshaped, train_labels, epochs=20, batch_size=128)

model.evaluate(test_images_reshaped, test_labels)

model.save("mnist_model.h5")

def ifi(output):
    if output == 0:
        return "this is zero"
    elif output == 1:
        return "tis is one"
    elif output == 2:
        return "tis is two"
    elif output == 3:
        return "tis is three"
    elif output == 4:
        return "tis is four"
    elif output == 5:
        return "tis is five"
    elif output == 6:
        return "tis is six"
    elif output == 7:
        return "tis is seven"
    elif output == 8:
        return "tis is eight"
    elif output == 9:
        return "tis is nine"
    else:
        return "invalid output"

def image_prediction(img):
    image = Image.open(img)
    img = image.convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img)
    
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        
    img_array = img_array.astype('float') / 255
    img_array = img_array.reshape(1, 784)
    output = np.argmax(model.predict(img_array))
    return ifi(output)
