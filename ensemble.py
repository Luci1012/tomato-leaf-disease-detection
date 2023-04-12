'''

To combine VGG16 with MobileNet using ensemble learning and voting on the Plant Village dataset, you can follow the steps below:
1.	Import the necessary libraries and modules.
2.	Load the Plant Village dataset and split it into training and testing sets.
3.	Create instances of the VGG16 and MobileNet models and load their pre-trained weights.
4.	Remove the top layers of both models and freeze the weights of their remaining layers.
5.	Create a new model by combining the output layers of the VGG16 and MobileNet models.
6.	Add a new fully connected layer with a Softmax activation function to the combined model.
7.	Train the combined model using the training set and validate it using the testing set.
8.	Use the trained model to make predictions on the testing set.
9.	Use voting to combine the predictions of the VGG16 and MobileNet models.
10.	Evaluate the accuracy of the ensemble model.

'''

# Step 1: Import the necessary libraries and modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNet
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# Step 2: Load the Plant Village dataset and split it into training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator_vgg = train_datagen.flow_from_directory(
    './train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

train_generator_mobilenet = train_datagen.flow_from_directory(
    './train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator_vgg = test_datagen.flow_from_directory(
    './val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator_mobilenet = test_datagen.flow_from_directory(
    './val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Step 3: Create instances of the VGG16 and MobileNet models and load their pre-trained weights
img_shape = (224, 224, 3)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=img_shape)

# Step 4: Remove the top layers of both models and freeze the weights of their remaining layers
for layer in vgg16.layers:
    layer.trainable = False
vgg16_output = vgg16.layers[-1].output

for layer in mobilenet.layers:
    layer.trainable = False
mobilenet_output = mobilenet.layers[-1].output

# Step 5: Create a new model by combining the output layers of the VGG16 and MobileNet models
combined_output = concatenate([vgg16_output, mobilenet_output])

# Step 6: Add a new fully connected layer with a Softmax activation function to the combined model
dense_output = Dense(256, activation='relu')(combined_output)
dense_output = Dropout(0.5)(dense_output)
dense_output = Dense(38, activation='softmax')(dense_output)

# # Step 7: Train the combined model using the training set and validate it using the testing set
# model = Model(inputs=[vgg16.input, mobilenet.input], outputs=dense_output)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     [train_generator, train_generator],
#     epochs=10,
#     validation_data=([test_generator, test_generator])
# )

# Step 7: Train the combined model using the training set and validate it using the testing set
model = Model(inputs=[vgg16.input, mobilenet.input], outputs=dense_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
steps_per_epoch = len(train_generator_vgg) # batch_size
validation_steps = len(test_generator_vgg) # batch_size

history = model.fit(
    x=[train_generator_vgg, train_generator_mobilenet],
    epochs=10,
    validation_data=([test_generator_vgg, test_generator_mobilenet]),
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Step 8: Use the trained model to make predictions on the testing set
y_pred_vgg16 = vgg16.predict(test_generator)
y_pred_mobilenet = mobilenet.predict(test_generator)
y_pred_combined = np.argmax(y_pred_vgg16 + y_pred_mobilenet, axis=1)

# Step 9: Use voting to combine the predictions of the VGG16 and MobileNet models
y_pred_vgg16 = np.argmax(y_pred_vgg16, axis=1)
y_pred_mobilenet = np.argmax(y_pred_mobilenet, axis=1)

y_pred_ensemble = []
for i in range(len(y_pred_vgg16)):
    votes = [y_pred_vgg16[i], y_pred_mobilenet[i]]
    y_pred_ensemble.append(max(set(votes), key=votes.count))

# Step 10: Evaluate the accuracy of the ensemble model
y_true = test_generator.classes
accuracy = accuracy_score(y_true, y_pred_ensemble)
print('Ensemble accuracy:', accuracy)

# In step 8, argmax() is a function that returns the index of the maximum value in a given array. In the context of machine learning, 
# argmax() is commonly used to obtain the predicted class label from a model's output probabilities.

# In step 9, voting is used to combine the predictions of the VGG16 and MobileNet models. 
# The argmax method is called on y_pred_vgg16 and y_pred_mobilenet to get the predicted class for each image in the testing set.

# Next, a for loop is used to iterate through each predicted class in y_pred_vgg16. 
# For each prediction, the class with the highest number of votes is determined by creating a list of the VGG16 and 
# MobileNet predictions for that image and using the max and count methods to find the most commonly occurring class in the list. 
# The resulting class is added to a new list called y_pred_ensemble.
