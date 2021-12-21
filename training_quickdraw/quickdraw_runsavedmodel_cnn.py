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


# In[8]:


model = tf.keras.models.load_model('save/qd_cnn_345_1000.h5')
model.summary()


# In[17]:


#Predict
filename = 'path_to_image'
predict_image(filename)


# In[ ]:




