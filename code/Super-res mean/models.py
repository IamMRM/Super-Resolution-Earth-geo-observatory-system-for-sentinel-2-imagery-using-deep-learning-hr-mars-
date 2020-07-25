import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Activation
import numpy as np
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from architectures import UNet
from segmentation_models import Unet as seg_Unet
import keras.backend as K


def load_nlcd_stats():
    nlcd_means = np.loadtxt("nlcd_mu.txt")
    #nlcd_means = nlcd_means / np.maximum(0, nlcd_means).sum(axis=1, keepdims=True)#normalize krne ke liye
    nlcd_vars = np.loadtxt("nlcd_std.txt")
    nlcd_vars[nlcd_vars < 0.0001] = 0.0001
    #nlcd_class_weights = np.ones((11,))
    nlcd_class_weights = np.array([1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0
    ])
    return nlcd_class_weights, nlcd_means, nlcd_vars


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

def jaccard_loss(y_true, y_pred, smooth=0.001, num_classes=10):
    intersection = y_true * y_pred
    sum_ = y_true + y_pred
    jac = K.sum(intersection + smooth, axis=(0,1,2)) / K.sum(sum_ - intersection + smooth, axis=(0,1,2))
    return (1.0 - K.sum(jac) / num_classes)

def accuracy(y_true, y_pred):
    num = K.sum(K.cast(K.equal(K.argmax(y_true[:,:,:,1:], axis=-1), K.argmax(y_pred[:,:,:,1:], axis=-1)), dtype="float32") * y_true[:,:,:,0])
    denom = K.sum(y_true[:,:,:,0])
    return num / (denom + 1) # make sure we don't get divide by zero


def hr_loss(boundary=0):
    '''The first channel of y_true should be all 1's if we want to use hr_loss, or all 0's if we don't want to use hr_loss
    '''

    def loss(y_true, y_pred):
        return categorical_crossentropy(y_true[:, :, :, 1:], y_pred[:, :, :, 1:]) * y_true[:, :, :, 0]

    return loss


def sr_loss(nlcd_class_weights, nlcd_means, nlcd_vars, boundary=0):
    '''Calculate superres loss according to ICLR paper'''
    def ddist(prediction, c_interval_center, c_interval_radius):
        return K.relu(K.abs(prediction - c_interval_center) - c_interval_radius)

    def loss(y_true, y_pred):
        super_res_crit = 0
        mask_size = K.expand_dims(K.sum(y_true, axis=(1, 2, 3)) + 10, -1)  # shape 16x1

        for nlcd_idx in range(nlcd_class_weights.shape[0]):
            c_mask = K.expand_dims(y_true[:, :, :, nlcd_idx], -1)  # shape 16x240x240x1
            c_mask_size = K.sum(c_mask, axis=(1, 2)) + 0.000001  # shape 16x1

            c_interval_center = nlcd_means[nlcd_idx]  # shape 5,
            c_interval_radius = nlcd_vars[nlcd_idx]  # shape 5,

            masked_probs = y_pred * c_mask  # (16x240x240x5) * (16x240x240x1) --> shape (16x240x240x5)

            # Mean mean of predicted distribution
            mean = K.sum(masked_probs, axis=(1, 2)) / c_mask_size  # (16x5) / (16,1) --> shape 16x5

            # Mean var of predicted distribution
            var = K.sum(masked_probs * (1. - masked_probs), axis=(1, 2)) / (
                        c_mask_size * c_mask_size)  # (16x5) / (16,1) --> shape 16x5

            c_super_res_crit = K.square(
                ddist(mean, c_interval_center, c_interval_radius))  # calculate numerator of equation 7 in ICLR paper
            c_super_res_crit = c_super_res_crit / (
                        var + (c_interval_radius * c_interval_radius) + 0.000001)  # calculate denominator
            c_super_res_crit = c_super_res_crit + K.log(var + 0.03)  # calculate log term
            c_super_res_crit = c_super_res_crit * (c_mask_size / mask_size) * nlcd_class_weights[
                nlcd_idx]  # weight by the fraction of NLCD pixels and the NLCD class weight

            super_res_crit = super_res_crit + c_super_res_crit  # accumulate

        super_res_crit = K.sum(super_res_crit, axis=1)  # sum superres loss across highres classes
        return super_res_crit

    return loss


def unet(img_shape, num_classes, optimizer, loss):
    i, o = UNet(img_shape, dims=[64, 32, 32, 32, 32], out_ch=num_classes)
    o = Activation("softmax", name="outputs_hr")(o)
    return make_model(i, o, optimizer, loss)

def make_model(inputs, outputs, optimizer, loss):
    if loss == "superres":
        outputs_sr = Lambda(lambda x: x, name="outputs_sr")(outputs)
        model = Model(inputs=inputs, outputs=[outputs, outputs_sr])
    else:
        model = Model(inputs=inputs, outputs=outputs)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "superres":
        nlcd_class_weights, nlcd_means, nlcd_vars = load_nlcd_stats()
        model.compile(
            optimizer=optimizer,
            loss={
                "outputs_hr": "categorical_crossentropy",
                "outputs_sr": sr_loss(nlcd_class_weights, nlcd_means, nlcd_vars)
            },
            loss_weights={
                "outputs_hr": 100,
                "outputs_sr": 1
            },
            metrics={
                "outputs_hr": [iou_coef,dice_coef]
            }
        )
    else:
        print("Loss function not specified, model not compiled")
    return model