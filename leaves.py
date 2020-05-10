import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.image as image
import pathlib

# RGB to Grayscale function can also be done with openCV or PIL libraries
# The function is here done as weighted product of the original image

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#
# The leaves are of type 0,1 or 2 ... the names have been kept to read the type
#

def getImageType(imgName):
    imgType = 0
    if "One" in imgName:
        imgType = 1
    elif "Two" in imgName :
        imgType = 2
    return imgType

#
# Function which takes in two lists and a folder path, reads the images and labels, after converting to RGB
#
def readImages(images,labels,path):
    data_dir = pathlib.Path(path)
    imgsRead = list(data_dir.glob('*.jpg'))
    for imgName in imgsRead:
        im = image.imread(imgName)
        images.append(rgb2gray(im))
        labels.append(getImageType(imgName.name))
    return images,labels
#
# Load images into arrays here
#
print("Load leaf images....")

train_images = []
train_labels = []
test_images = []
test_labels = []

# Modify the path to where the leaves images are stored


train_images, train_labels = readImages(train_images,train_labels, "/users/hari/Leaves/0")
train_images, train_labels = readImages(train_images, train_labels, "/users/hari/Leaves/1")
train_images, train_labels = readImages(train_images, train_labels, "/users/hari/Leaves/2")
test_images, test_labels = readImages(test_images, test_labels, "/users/hari/Leaves/Test")

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

print("images loaded for training:", len(train_images))
print("images loaded for testing:", len(test_images))

#
# Gray scale images need to be scaled down to a number between 0 and 1 for usage in tensorflow
#

train_images = train_images / 255
test_images = test_images / 255

#
# Boiler plate code for flattening, defining layers, function etc
# Defines and compiles the model
# Finally fits using the train data and labels
#

model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(640, 480)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3),
])

model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

print("Model created")

#
# Predict the test images and compare it against the expected.
# Outputs "Predicted" vs "Expected"
#

pred = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = pred.predict(test_images)
print("Predicted vs Expected")
for i in range(len(test_images)):
    print(np.argmax(predictions[i]), " vs ", test_labels[i])
