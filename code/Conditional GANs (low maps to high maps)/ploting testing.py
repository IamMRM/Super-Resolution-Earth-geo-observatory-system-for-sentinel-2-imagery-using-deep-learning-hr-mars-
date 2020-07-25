#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID= '0'
os.environ["CUDA_VISIBLE_DEVICES"] =GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


# In[2]:


import keras
# ^^^ pyforest auto-imports - don't write above this line
# example of loading a pix2pix model and using it for image to image translation
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import numpy as np


# In[3]:


def normalize(array):
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min))


# In[4]:


# ^^^ pyforest auto-imports - don't write above this line
# load and prepare training images
def load_real_samples(training_patches):
    # load compressed arrays
    X1= np.zeros((len(training_patches),256,256,3))
    X2= np.zeros((len(training_patches),256,256,8))
    for i,(filename, state) in enumerate(training_patches):
        data = np.load(filename)
        # unpack arrays
        redn = normalize(data[:, :, 0])
        greenn = normalize(data[:, :, 1])
        bluen = normalize(data[:, :, 2])
        #infraredn = normalize(data[:, :, 3])
        X1[i] = np.dstack((redn, greenn, bluen))
        #X2[i] = np.copy(np.expand_dims(data[:, :, 5],axis=2))
        y_train_hr = np.copy(data[:, :, 5])
        y_train_hr[y_train_hr == 1] = 0
        y_train_hr[y_train_hr == 2] = 1
        y_train_hr[y_train_hr == 4] = 2
        y_train_hr[y_train_hr == 5] = 3
        y_train_hr[y_train_hr == 6] = 4
        y_train_hr[y_train_hr == 7] = 5
        y_train_hr[y_train_hr == 9] = 6
        y_train_hr[y_train_hr == 10] = 7
        y_train_hr = keras.utils.to_categorical(y_train_hr, 8)
        X2[i] = y_train_hr
    return [X1, X2]


# In[5]:


# ^^^ pyforest auto-imports - don't write above this line
data_dir = '../SEN12MS/processed/'
training_states = ['ROIs13_high_roshan_s2']
validation_states = ['ROIs13_high_roshan_s2']
training_patches=[]
for state in training_states:
    fn= os.path.join(data_dir,'%s_train_patches/' %state)
    #print(fn)
    with os.scandir(fn) as entries:
        for entry in entries:
            #print(entry.name)
            with os.scandir(fn+entry.name) as folders:
                for folder in folders:
                    #print(folder.name)
                    with os.scandir(fn+entry.name+'/'+folder.name) as files:
                        for file in files:
                            #print(file.name)
                            training_patches.append((os.path.join(fn+entry.name+'/'+folder.name+'/',file.name),state))
#print(training_patches)

validation_patches=[]
for state in validation_states:
    fn= os.path.join(data_dir,'%s_val_patches/' %state)
    #print(fn)
    with os.scandir(fn) as entries:
        for entry in entries:
            #print(entry.name)
            with os.scandir(fn+entry.name) as folders:
                for folder in folders:
                    #print(folder.name)
                    with os.scandir(fn+entry.name+'/'+folder.name) as files:
                        for file in files:
                            #print(file.name)
                            validation_patches.append((os.path.join(fn+entry.name+'/'+folder.name+'/',file.name),state))
#print(validation_patches)


# In[6]:


# load dataset
[X1, X2] = load_real_samples(validation_patches)


# In[8]:


model = load_model('model_172980.h5')


# In[23]:


#selecting a random exmaple
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]


# In[24]:


print(ix)
print(src_image.shape)


# In[25]:


gen_image = model.predict(src_image)


# In[12]:


lab2rgb = {0:[0./255, 128./255, 0./255],        #dark-green         ->          Forest
           1:[205./255, 133./255, 63./255],     #peruvian-brown     ->          Dense-Shrublands
           2:[224./255, 216./255, 202./255],    #savana-oaks        ->          Woody-Savanas
           3:[102./255, 102./255, 25./255],    #dark-olive         ->          Grasslands
           4:[210./255, 209./255, 205./255],   #concrete           ->          Urban
           #5:[229./255, 229./255, 178./255],   #pale-olive         ->          Vegetation
           5:[255./255, 255./255, 255./255],   #white              ->          Snow-and-ice
           6:[131./255, 117./255, 96./255],    #barren-paint       ->          Barren
           7:[135./255, 206./255, 250./255],   #light-blue         ->          Water-bodies
          }
rgb = np.ones((256,256,3))


# In[13]:


# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
    #images = vstack((src_img, gen_img, tar_img))
    # scale from [-1,1] to [0,1]
    #images = (images + 1) / 2.0
    #titles = ['Source', 'Generated', 'Expected']
    # plot images row by row
    
    n_samples =1
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(src_img[i])
    # plot generated target image
    for i in range(n_samples):
        label = np.argmax(gen_img[i], axis=-1)
        for p in range(256):
            for q in range(256):
                rgb[p,q,:] = np.expand_dims(np.array(lab2rgb[label[p,q]]),axis=0)
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(rgb)
    # plot real target image
    for i in range(n_samples):
        label = np.argmax(tar_img[i], axis=-1)
        for p in range(256):
            for q in range(256):
                rgb[p,q,:] = np.expand_dims(np.array(lab2rgb[label[p,q]]),axis=0)
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(rgb)


# In[26]:


plot_images(src_image, gen_image, tar_image)