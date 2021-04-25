from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import io
from tensorflow.python.keras.applications.densenet import layers

# Re sizing the image file
IMAGE_SIZE = [224, 224]

# Assigning the training and testing image data set
Training_set = 'Datasets/Training'
Test_set = 'Datasets/Testing'


# Adding preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',include_top=False)

# Freezing the layers of the model because we don't want to train them
for layer in vgg.layers:
  layer.trainable = False

# To get all the folders/data from inside the mentioned folder
folders = glob('Datasets/Training/*')

# Adding the flatten layer
x = Flatten()(vgg.output)
y = Dropout(.2,input_shape=IMAGE_SIZE + [3])(x)

# Making the prediction
prediction = Dense(len(folders), activation='softmax')(y)

# Creating new model based on the prediction
model = Model(inputs=vgg.input, outputs=prediction)
#print(model.summary())


# Compiling the new Model
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
'''
img_input = layers.Input(shape=(224, 224, 3))  # or IMAGE_SIZE + [3])

# Being the first layer input shape is given
# First convolution extracts 16 filters that are 3x3
# layers.Conv2D(filters, kernel_size, activation, input size)
x = layers.Conv2D(16, 3, activation='relu')(img_input) # Outputs filtered kernel, smaller than actual image

#  Max-pooling layer with a 2x2 window followed after convolution layer
x = layers.MaxPooling2D(2)(x) # Down-sampling input file to make more features assumption

# Second convolution extracts 32 filters that are 3x3
x = layers.Conv2D(32, 3, activation='relu')(x)

# Max-pooling again for features extraction
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.25)(x) # Dropout layer for over-fitting
# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.25)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(5, activation='sigmoid')(x)

# Create model:
model = Model(img_input, output)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
'''

# Doing some changes in the used image dataset
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) # reversing columns of pixels

test_datagen = ImageDataGenerator(rescale = 1./255)

# Getting the batches of images from the mentioned derectories used for training and testing purpose
training_set = train_datagen.flow_from_directory('Datasets/Training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Datasets/Testing',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# Fitting model
history = model.fit(
  training_set,
  validation_data=test_set,  # used testing set as validation set
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# Saving the model
model.save('Model6_using_Vgg16.h5')

# list all data in history
print(history.history.keys())

# Show history of accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Show history for validation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



















