import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

train_path = "C:/Users/DELL/Downloads/Plant-Leaf-Disease-Prediction-main/Plant-Leaf-Disease-Prediction-main/train"
test_path = "C:/Users/DELL/Downloads/Plant-Leaf-Disease-Prediction-main/Plant-Leaf-Disease-Prediction-main/val"

train_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_datagen = ImageDataGenerator(preprocessing_function=vgg_preprocess)
test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

vgg_model = VGG16(weights='imagenet', include_top=False)
for layer in vgg_model.layers:
    layer.trainable = False

mobilenet_model = MobileNet(weights='imagenet', include_top=False)
for layer in mobilenet_model.layers:
    layer.trainable = False

def predict_generator(model, generator):
    return model.predict_generator(generator, steps=len(generator), verbose=1)

vgg_train_pred = predict_generator(vgg_model, train_generator)
vgg_test_pred = predict_generator(vgg_model, test_generator)

mobilenet_train_pred = predict_generator(mobilenet_model, train_generator)
mobilenet_test_pred = predict_generator(mobilenet_model, test_generator)

train_pred = np.concatenate([vgg_train_pred, mobilenet_train_pred], axis=1)
test_pred = np.concatenate([vgg_test_pred, mobilenet_test_pred], axis=1)

ensemble_train_pred = np.argmax(train_pred, axis=1)
ensemble_test_pred = np.argmax(test_pred, axis=1)

ensemble_accuracy = np.mean(np.equal(ensemble_test_pred, test_generator.classes))
print('Ensemble accuracy:', ensemble_accuracy)
