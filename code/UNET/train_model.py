import sys
import shutil
import os
import time
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.optimizers import RMSprop
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


# In[ ]:


verbose = 1
output = 'output/'
name = 'experiment1'

data_dir = 'SEN12MS/processed/'
training_states = ['ROI2017_winter_s2', 'ROIs1970_fall_s2','ROIs1868_summer_s2','ROIs1158_spring_s2']
validation_states = ['ROI2017_winter_s2']

model_type = "unet"
batch_size = 16
learning_rate = 0.01
loss = "crossentropy"
log_dir = os.path.join(output, name)
#os.makedirs(log_dir)
assert os.path.exists(log_dir), "Output directory doesn't exist"

# In[ ]:


#Load data updated
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


for i in range(20000):
    np.random.shuffle(training_patches)
    np.random.shuffle(validation_patches)


training_steps_per_epoch = 300
validation_steps_per_epoch = 25

print("Number of training/validation steps per epoch: %d/%d" % (training_steps_per_epoch, validation_steps_per_epoch))


# In[ ]:


import models
optimizer = RMSprop(learning_rate)
model = models.unet((256,256,4), 8, optimizer, loss)
#model.summary()


# In[ ]:


import datagen
training_generator = datagen.DataGenerator(training_patches, batch_size, training_steps_per_epoch, 256, 256, 4)
validation_generator = datagen.DataGenerator(validation_patches, batch_size, validation_steps_per_epoch, 256, 256, 4)


# In[ ]:

model_checkpoint_callback = ModelCheckpoint(
        os.path.join(log_dir, "model_{epoch:02d}.h5"),
        verbose=verbose,
        save_best_only=True,
        save_weights_only=False,
        period=1
    )


results = model.fit_generator(
        training_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=100,#10**6
        verbose=verbose,
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch,
        max_queue_size=64,
        workers=4,
        use_multiprocessing=True,
        callbacks=[model_checkpoint_callback],
        initial_epoch=0
    )


# In[ ]:


model.save(os.path.join(log_dir, "final_model.h5"))

model_json = model.to_json()
with open(os.path.join(log_dir,"final_model.json"), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(log_dir, "final_model_weights.h5"))


# In[ ]:

#END
print(results.history)

with open('listfile.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in results)