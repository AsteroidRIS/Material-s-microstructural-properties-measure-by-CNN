#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('tensorflow version:',tf.__version__)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import PIL
print('PIL version:',PIL.__version__)
import os
import numpy as np
print('numpy version:',np.__version__)
import matplotlib
import matplotlib.pyplot as plt
print('matplotlib version:',matplotlib.__version__)


# In[2]:


# define the image size and batch size
batch_size = 32
img_height = 224
img_width = 224


# In[3]:


# preprocessing
train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)


# In[4]:


# spilting of dataset for trainig purpose
train_dataset = tf.keras.preprocessing.image_dataset_from_directory('D:/materials', 
                                            validation_split = 0.2,
                                            subset = 'training',
                                            seed = 123,
                                            image_size = (img_height, img_width),
                                            batch_size = batch_size)


# In[5]:


# spilting of dataset for testing purpose
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory('D:/materials', 
                                            validation_split = 0.2,
                                            subset = 'validation',
                                            seed = 123,
                                            image_size = (img_height, img_width),
                                            batch_size = batch_size)


# In[6]:


class_names_train = train_dataset.class_names
print(class_names_train)
class_names_validation = validation_dataset.class_names
print(class_names_validation)


# In[7]:


for image_batch, label_batch in train_dataset:
    print(image_batch.shape)
    print(label_batch.shape)
    break


# In[8]:


normalization_layers = layers.experimental.preprocessing.Rescaling(1/255)


# In[9]:


normalized_dataset = train_dataset.map(lambda x, y: (normalization_layers(x), y))
image_batch, labels_batch = next(iter(normalized_dataset))
first_image = image_batch[0]
#pixel value now b/w 0 - 1
print(np.min(first_image), np.max(first_image))


# In[10]:


num_classes = 2
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1/255, input_shape = (img_height, img_width, 3)),
    layers.Conv2D(32, 5, padding='same', activation = 'relu'),
    layers.MaxPooling2D( pool_size = (2,2), strides = (2,2)),
    layers.Conv2D(32, 3, padding='same', activation = 'relu'),
    layers.MaxPooling2D( pool_size = (2,2), strides = (2,2)),
    layers.Conv2D(32, 2, padding='same', activation = 'relu'),
    layers.MaxPooling2D( pool_size = (2,2), strides = (2,2)),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(num_classes)
    ])


# In[11]:


optimizers = tf.keras.optimizers.Adam(learning_rate = 0.0005)
model.compile(optimizers,
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
             metrics = ['accuracy'])


# In[12]:


model.summary()


# In[13]:


epochs = 30
history = model.fit(
train_dataset,
epochs = epochs
)


# In[14]:


model.evaluate(validation_dataset)


# In[ ]:




