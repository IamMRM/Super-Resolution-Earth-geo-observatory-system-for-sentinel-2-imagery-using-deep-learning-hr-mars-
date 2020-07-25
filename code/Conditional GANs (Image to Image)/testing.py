import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID= '0'
os.environ["CUDA_VISIBLE_DEVICES"] =GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import keras
from keras.models import load_model
from keras.optimizers import Adam
import keras.backend as K

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min))

# load and prepare testing images
def load_real_samples(testing_patches):
    # load compressed arrays
    X1= np.zeros((len(testing_patches),256,256,4))
    X2= np.zeros((len(testing_patches),256,256,8))
    for i,(filename, state) in enumerate(testing_patches):
        data = np.load(filename)
        # unpack arrays
        redn = normalize(data[:, :, 0])
        greenn = normalize(data[:, :, 1])
        bluen = normalize(data[:, :, 2])
        infraredn = normalize(data[:, :, 3])
        X1[i] = np.dstack((redn, greenn, bluen,infraredn))
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

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

data_dir = '../SEN12MS/processed/'
testing_states = ['ROIs13_high_roshan_s2']

testing_patches=[]
for state in testing_states:
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
                            testing_patches.append((os.path.join(fn+entry.name+'/'+folder.name+'/',file.name),state))
#Loading the data
testA, testB = load_real_samples(testing_patches)

opt = Adam(lr=0.0002, beta_1=0.5)
abv= load_model("final_model.h5")
abv.compile(optimizer= opt,loss= 'categorical_crossentropy',metrics = [iou_coef,dice_coef])
mod = abv.evaluate(x=testA,y=testB,batch_size=1)
print(mod)