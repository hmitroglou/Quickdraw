#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import asarray



from PIL import Image


# In[15]:


model = tf.keras.models.load_model('save/qd_0')
model.summary()


# In[19]:


labels = ['airplane', 'axe', 'basketball','hexagon','couch']

#Convert png to 28x28 array
image = Image.open('dataset/airplane3.png')
image_resize = image.resize((28,28))
data_resize = asarray(image_resize)
data_resize = abs(data_resize-255.)/255.
data_resize = data_resize[:, :, 0]
data_resize = np.expand_dims(data_resize, axis=0)

#Predict png
predictions = model.predict(data_resize)
print(labels[np.argmax(predictions)])
print(data_resize.shape)
plt.imshow(image_resize)
plt.show()


# In[ ]:




