import keras.backend as K
from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.layers import Lambda, Activation
from keras.losses import categorical_crossentropy


def conv_bn_relu(m, dim):
    x = Conv2D(dim, 3, activation=None, padding='same')(m)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet_block(m, dim, res=False):
    x = m
    for i in range(3):
        x = conv_bn_relu(x, dim)
    return Concatenate()([m, x]) if res else x

def dense_block(m, dim):
    x = m
    outputs = [x]
    for i in range(3):
        conv = conv_bn_relu(x, dim)
        x = Concatenate()([conv, x])
        outputs.append(conv)
    return Concatenate()(outputs)

def level_block_fixed_dims(m, dims, depth, acti, do, bn, mp, up, res, dense=False):
    max_depth = len(dims)-1
    dim = dims[max_depth-depth]
    if depth > 0:
        n = dense_block(m, dim) if dense else unet_block(m, dim, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block_fixed_dims(m, dims, depth-1, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = dense_block(n, dim) if dense else unet_block(n, dim, res)
    else:
        m = dense_block(m, dim) if dense else unet_block(m, dim, res)
    return m

def UNet(img_shape, dims=[32, 64, 128, 256, 128], out_ch=1, activation='relu', dropout=False, batchnorm=True, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block_fixed_dims(i, dims, len(dims)-1, activation, dropout, batchnorm, maxpool, upconv, residual, dense=False)
    o = Conv2D(out_ch, 1, activation=None, name="logits")(o)
    return i, o

i,o = UNet(img_shape=(256,256,4),dims=[64, 32, 32, 32, 32],out_ch= 8)
o = Activation("softmax", name="outputs_hr")(o)
model = Model(inputs=i, outputs=o)
model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy"], optimizer="Adam")
model.summary()