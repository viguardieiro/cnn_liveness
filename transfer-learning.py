#!/usr/bin/env python
# coding: utf-8

# In[16]:


from imutils import paths
import os
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications import VGG16 
from keras.preprocessing import image
from keras.models import model_from_json
from keras import models,layers,optimizers
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input
from keras.layers import  AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge
from keras.layers import Input, Dense, Reshape, Activation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
import cv2
import numpy as np


# In[26]:


# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 8
EPOCHS = 10

imagePaths = list(paths.list_images('dataset'))
data = []
labels = []


# In[9]:


for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(label)


# In[19]:


labels


# In[17]:


# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0


# In[18]:


# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)


# In[20]:


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)


# In[21]:


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")


# In[22]:


image_size=224
vgg_base = VGG16(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3))


# In[23]:


#initiate a model
model = models.Sequential()

#Add the VGG base model
model.add(vgg_base)

#Add new layers
model.add(layers.Flatten())

model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# In[24]:


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint("Checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                             monitor='val_loss', 
                             verbose=0, 
                             save_best_only=False, 
                             save_weights_only=False, 
                             mode='auto', 
                             period=1)


# In[27]:


model.fit(trainX,
          trainY,
          epochs=12, 
          batch_size=BS,
          validation_data=(testX, testY))


# In[ ]:


# Evaluate the model
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))


# In[ ]:


print("[INFO] Plotting stats")
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

