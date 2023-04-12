import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Plant Village dataset and split it into training and testing sets
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    './val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Step 2: Create an instance of the VGG16 model and load its pre-trained weights
img_shape = (224, 224, 3)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)

# Step 3: Remove the top layers of the VGG16 model and freeze the weights of its remaining layers
for layer in vgg16.layers:
    layer.trainable = False
vgg16_output = vgg16.layers[-1].output

# Step 4: Create a new model that outputs the VGG16 feature vectors
feature_extractor = Model(inputs=vgg16.input, outputs=vgg16_output)

# Step 5: Extract features from the training and testing sets using the VGG16 feature extractor
train_features = feature_extractor.predict(train_generator)
test_features = feature_extractor.predict(test_generator)

# Step 6: Flatten the feature vectors
train_features_flat = np.reshape(train_features, (train_features.shape[0], -1))
test_features_flat = np.reshape(test_features, (test_features.shape[0], -1))

# Step 7: Train a Random Forest classifier on the flattened feature vectors
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_features_flat, train_generator.classes)

# Step 8: Use the trained model to make predictions on the testing set
y_pred_rf = rf.predict(test_features_flat)

# Step 9: Evaluate the accuracy of the Random Forest classifier
accuracy = accuracy_score(test_generator.classes, y_pred_rf)
print('Random Forest accuracy:', accuracy)
