#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
# ^^^ pyforest auto-imports - don't write above this line
"""The Pix2Pix model is a type of conditional GAN, or cGAN, where the 
generation of the output image is conditional on an input, in this case,
a source image. The discriminator is provided both with a source image 
and the target image and must determine whether the target is a plausible 
transformation of the source image."""



import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID= '0'
os.environ["CUDA_VISIBLE_DEVICES"] =GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


# In[3]:


# example of pix2pix gan for satellite to map image-to-image translation
import numpy as np
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot


# In[4]:


#discriminator
def define_discriminator(image_shape=(256,256,8)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_src_image = Input(shape=(256,256,8))
    # target image input
    in_target_image = Input(shape=(256,256,8))
    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    model.summary()
    return model


# In[5]:


__ = define_discriminator()


# In[4]:


from keras import backend as K
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


# In[7]:


# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model
def define_generator(image_shape=(256,256,8)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(8, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('softmax')(g)#-1 se +1 ke liye tanh kia h
    # define model
    model = Model(in_image, out_image)
    model.summary()
    return model


# In[8]:


_= define_generator()


# In[9]:


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=(256,256,8))
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt, loss_weights=[1,100],metrics=[iou_coef,dice_coef])
    return model


# In[5]:


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


# In[ ]:


def normalize(array):
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min))


# In[5]:


# load and prepare training images
def load_real_samples(training_patches):
    # load compressed arrays
    X1= np.zeros((len(training_patches),256,256,8))
    X2= np.zeros((len(training_patches),256,256,8))
    for i,(filename, state) in enumerate(training_patches):
        data = np.load(filename)
        # unpack arrays
        y_train_nlcd=np.copy(data[:, :, 4])
        y_train_nlcd[y_train_nlcd == 1] = 0# forest
        y_train_nlcd[y_train_nlcd == 2] = 0
        y_train_nlcd[y_train_nlcd == 3] = 0
        y_train_nlcd[y_train_nlcd == 4] = 0
        y_train_nlcd[y_train_nlcd == 5] = 0
        y_train_nlcd[y_train_nlcd == 6] = 1  # shrublands
        y_train_nlcd[y_train_nlcd == 7] = 1
        y_train_nlcd[y_train_nlcd == 8] = np.random.choice(np.arange(0, 8), p=[0.15, 0.05, 0.15, 0, 0, 0.15, 0.5, 0])#savannas
        y_train_nlcd[y_train_nlcd == 9] = np.random.choice(np.arange(0, 8), p=[0.15, 0.05, 0.15, 0, 0, 0.15, 0.5, 0])
        y_train_nlcd[y_train_nlcd == 10] = 2  # grassland
        y_train_nlcd[y_train_nlcd == 11] = 3  # wetlands
        y_train_nlcd[y_train_nlcd == 12] = 4  # croplands
        y_train_nlcd[y_train_nlcd == 14] = 4  # cropland
        y_train_nlcd[y_train_nlcd == 13] = 5  # builtup
        y_train_nlcd[y_train_nlcd == 15] = 7  # ice
        y_train_nlcd[y_train_nlcd == 17] = 7  # water
        y_train_nlcd[y_train_nlcd == 16] = 6  # barren
        y_train_nlcd = keras.utils.to_categorical(y_train_nlcd, 8)
        X1[i]=y_train_nlcd
        
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


# In[12]:


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


# In[13]:


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


# In[14]:


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # plot real source image    
    for i in range(n_samples):
        label = np.argmax(X_realA[i], axis=-1)
        for p in range(256):
            for q in range(256):
                rgb[p,q,:] = np.expand_dims(np.array(lab2rgb[label[p,q]]),axis=0)
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(rgb)
    # plot generated target image
    for i in range(n_samples):
        label = np.argmax(X_fakeB[i], axis=-1)
        for p in range(256):
            for q in range(256):
                rgb[p,q,:] = np.expand_dims(np.array(lab2rgb[label[p,q]]),axis=0)
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(rgb)
    # plot real target image
    for i in range(n_samples):
        label = np.argmax(X_realB[i], axis=-1)
        for p in range(256):
            for q in range(256):
                rgb[p,q,:] = np.expand_dims(np.array(lab2rgb[label[p,q]]),axis=0)
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        pyplot.imshow(rgb)
    # save plot to file
    filename1 = 'plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# In[15]:


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=200, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _,_,_,g_iou,g_dice = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f] g_iou[%.3f] g_dice[%.3f]' % (i+1, d_loss1, d_loss2, g_loss,g_iou, g_dice))
        # summarize model performance
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)


# In[6]:


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


# In[ ]:


print(training_patches[0])


# In[17]:


dataset = load_real_samples(training_patches)


# In[18]:


# load image data
#print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)