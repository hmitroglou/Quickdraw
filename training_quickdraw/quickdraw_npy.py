#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

import numpy as np
from numpy import asarray

import matplotlib.pyplot as plt 
from matplotlib import image

import math as m
from PIL import Image
import random

import requests as req
import csv
import os
print(os.getcwd())

import copy

print(tf.__version__)


# In[2]:


def download_npy(categories,number_of_samples):

    number_of_categories = len(categories)
    data = np.array([], dtype=np.int64).reshape(0,784)
    for i in range(number_of_categories):
        filename = categories[i][0] + '.npy'
        filename = filename.replace(" ","%20")
        
        url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'+filename
        
        print(i+1,'/',number_of_categories,' ','download ',   filename.replace("%20",""))
        r = req.get(url, allow_redirects=True)
        #filename = os.path.basename(url)
        #filename = filename.replace("%20","")
        open(filename, 'wb').write(r.content)
        
        data = np.vstack([data,load_data(filename,number_of_samples)])
        
        os.remove(filename)
    return data

def load_data(name,n):
    filename = name
    label = name
    data = np.load(filename)
    #data = np.ndarray.reshape(data,len(data),28,28)
    return data[0:n,:]

#randomize data and labels
def shuff(data,label):
    
    s = np.shape(data)
    n = s[0]
    
    l = len(label)
    label_new = np.zeros(l)
    
    if len(2*s)==2:
        d = 1
        m = 1
        data_new = np.zeros((n,m))

    elif len(2*s)==4:
        d = 2
        m = s[1]
        data_new = np.zeros((n,m))
    
    orderid = random.sample(range(n), n) 
    
    for i in range(n):
        data_new[i] = data[orderid[i]]
        label_new[i] = int(label[orderid[i]])
        
    #del data, label
    return data_new, label_new

def download_and_save(number_of_categories,number_of_samples):

    x = [random.randint(0, 345) for p in range(0, number_of_categories)]

    #load categories
    categories = open("categories.txt",'r')
    reader = csv.reader(categories)
    categories = [row for row in reader]
    categories = [categories[row] for row in x]
    
    d = download_npy(categories,number_of_samples)
    
    #save Data
    filename = 'dataset/data_{}_{}.csv'.format(number_of_categories,number_of_samples)
    np.savetxt(filename, d, delimiter=',')
    
    #save Categories
    filename = 'dataset/cat_{}_{}.csv'.format(number_of_categories,number_of_samples)
    with open(filename, 'w') as f: 
        write = csv.writer(f) 
        write.writerows(categories)
    return d, categories
    
def data_from_file(number_of_categories,number_of_samples):
    
    filename = 'dataset/data_{}_{}.csv'.format(number_of_categories,number_of_samples)
    d = np.loadtxt(filename, delimiter=',')
    filename = 'dataset/cat_{}_{}.csv'.format(number_of_categories,number_of_samples)
    categories = open(filename,'r')
    reader = csv.reader(categories)
    categories = [row for row in reader]
    
    return d, categories

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


# In[20]:


ncat = 5
nsam = 5000

##load data
d, cats = download_and_save(ncat,nsam)
#d, cats = data_from_file(ncat,nsam)

cat_id = np.repeat(range(ncat),nsam)

print('Categories: ', cats)
print(np.shape(d),' ',np.shape(cat_id))


# # FFNN

# In[21]:


#Shuffle data
data,cat_id = shuff(d,cat_id)

#reshape data into 28x28
data = np.reshape(data,(len(data),28,28))
print(np.shape(data),' ',np.shape(cat_id))


# In[22]:


#split data
training = 0.8
test = 1.-training

x_train = data[0:m.floor(training*len(data))]/ 255.0
y_train = cat_id[0:m.floor(training*len(cat_id))]

x_test = data[m.ceil(training*len(data)):len(data)]/ 255.0
y_test = cat_id[m.ceil(training*len(cat_id)):len(cat_id)]

print('Train-Set: ','Samples',np.shape(x_train)[0],'/ Labels', np.shape(y_train)[0])
print('Test-Set: ','Samples',np.shape(x_test)[0],'/ Labels', np.shape(y_test)[0])


# In[25]:


#TEST Show
i=random.randint(1,len(x_train))
plt.imshow(x_train[i])
print(cats[int(cat_id[i])],i)


# In[26]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(ncat),
  tf.keras.layers.Softmax()
])

#define lossfunction
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Compile: set optimizer, lossfunction, error metric
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#train the model
history = model.fit(x_train, y_train,batch_size = 32, epochs=10)

#test the model
print('Test')
model.evaluate(x_test,  y_test, verbose=2);
predictions = model.predict(x_test)


# In[30]:


#TEST Show
i=random.randint(1,len(x_test))
plt.imshow(x_test[i])
print(cats[np.argmax(predictions[i])])


# In[ ]:




