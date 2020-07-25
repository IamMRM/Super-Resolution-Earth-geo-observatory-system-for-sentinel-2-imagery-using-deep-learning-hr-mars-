#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID= '0'
os.environ["CUDA_VISIBLE_DEVICES"] =GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


# In[2]:


import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger


# In[3]:


verbose = 1
output = 'output/'
name = 'experiment1'

data_dir = '../SEN12MS/processed/'
training_states = ['ROI2017_winter_s2', 'ROIs1970_fall_s2','ROIs1868_summer_s2','ROIs1158_spring_s2']
validation_states = ['ROIs13_high_roshan_s2']
superres_states = []


# In[4]:


model_type = "unet"
batch_size = 4
learning_rate = 0.001
loss = "superres"
log_dir = os.path.join(output, name)
#os.makedirs(log_dir)
assert os.path.exists(log_dir), "Output directory doesn't exist"


# In[5]:


training_patches=[]
for state in training_states:
    fn= os.path.join(data_dir,'%s_train_patches/' %state)
    with os.scandir(fn) as entries:
        for entry in entries:
            with os.scandir(fn+entry.name) as folders:
                for folder in folders:
                    with os.scandir(fn+entry.name+'/'+folder.name) as files:
                        for file in files:
                            training_patches.append((os.path.join(fn+entry.name+'/'+folder.name+'/',file.name),state))
#print(training_patches)

validation_patches=[]
for state in validation_states:
    fn= os.path.join(data_dir,'%s_val_patches/' %state)
    with os.scandir(fn) as entries:
        for entry in entries:
            with os.scandir(fn+entry.name) as folders:
                for folder in folders:
                    with os.scandir(fn+entry.name+'/'+folder.name) as files:
                        for file in files:
                            validation_patches.append((os.path.join(fn+entry.name+'/'+folder.name+'/',file.name),state))
#print(validation_patches)


# In[6]:


csv_logger = CSVLogger('log.csv', append=True, separator=';')

for i in range(2000):
    np.random.shuffle(training_patches)
    np.random.shuffle(validation_patches)


# In[7]:


training_steps_per_epoch = 300
validation_steps_per_epoch = 39

print("Number of training/validation steps per epoch: %d/%d" % (training_steps_per_epoch, validation_steps_per_epoch))


# In[8]:


import models
optimizer = Adam(learning_rate)
model = models.unet((256,256,4), 8, optimizer, loss)


# In[9]:


import datagen
import datagen_diff

training_generator = datagen.DataGenerator(training_patches, batch_size, training_steps_per_epoch, 256, 256, 4)
validation_generator = datagen_diff.DataGenerator(validation_patches, batch_size, validation_steps_per_epoch, 256, 256, 4)


# In[10]:


#learning_rate_callback = LearningRateScheduler(utils.schedule_stepped, verbose=verbose)
model_checkpoint_callback = ModelCheckpoint(
        os.path.join(log_dir, "model_{epoch:02d}.h5"),
        verbose=verbose,
        save_best_only=True,
        save_weights_only=False,
        period=1
    )


# In[11]:


results = model.fit_generator(
        training_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=150,#10**6
        verbose=verbose,
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch,
        max_queue_size=64,
        workers=4,
        use_multiprocessing=True,
        callbacks=[model_checkpoint_callback,csv_logger],
        initial_epoch=0
    )


model.save(os.path.join(log_dir, "final_model.h5"))

model_json = model.to_json()
with open(os.path.join(log_dir,"final_model.json"), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(log_dir, "final_model_weights.h5"))

#END
print(results.history)


# In[ ]:




