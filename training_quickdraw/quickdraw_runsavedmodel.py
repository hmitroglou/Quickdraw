#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import asarray
import csv
from sys import getsizeof

from PIL import Image


# In[2]:


def predict_image(path_image):
    image = Image.open(path_image)

    image_resize = image.resize((28,28))
    image_small= asarray(image_resize)
    image_small = abs(image_small-255.)/255.
    image_small = image_small[:, :, 0]
    image_small = np.expand_dims(image_small, axis=0)
    
    predictions = model.predict(image_small)
    print(cats[np.argmax(predictions)])
    print(image_small.shape)
    plt.imshow(image_resize)
    plt.show()


# In[3]:


model = tf.keras.models.load_model('save/qd_cnn_345_1000.h5')
model.summary()


# In[17]:


#Predict
labels = open("categories.txt",'r')
reader = csv.reader(labels)
labels = [row for row in reader]

image = Image.open('dataset/car.png')
data_resize = image.resize((28,28))
data_resize = asarray(data_resize)
np.shape(data_resize)
data4 = np.reshape(data_resize,(-1,28,28,4))
data4_ = data_resize.reshape(-1, 28,28, 1)
np.shape(data4)
np.shape(data4_)
plt.imshow(data4[0])
p = model.predict(data4_)
print(labels[np.argmax(p)])


# In[9]:


labels = open("categories.txt",'r')
reader = csv.reader(labels)
labels = [row for row in reader]

#Convert png to 28x28 array
image = Image.open('dataset/star.png')

image_resize = image.resize((28,28))
data_resize = asarray(image_resize)
data_resize = abs(data_resize-255.)/255.
data_resize = np.reshape(data_resize,(28,28,1))
data_resize = data_resize[:, :, 0]
data_resize = np.expand_dims(data_resize, axis=0)

#Predict png
predictions = model.predict(data_resize)
print(labels[np.argmax(predictions)])
print(data_resize.shape)
plt.imshow(image_resize)
plt.show()


# In[ ]:




