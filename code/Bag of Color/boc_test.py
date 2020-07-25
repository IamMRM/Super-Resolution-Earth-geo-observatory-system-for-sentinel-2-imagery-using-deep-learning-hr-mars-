# python3
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MiniBatchKMeans
import sklearn
import os
from sklearn.svm import SVC
import pickle
import matplotlib
import keras
from keras import backend as K
from PIL import Image


def makeVocabulary(folder_name, batch_size=10):
    all_pixels = []
    total_images = 0
    kmeans = MiniBatchKMeans(n_clusters=50, random_state=0, max_iter=300, n_init=10)
    for season in os.listdir(folder_name + "/"):
        for filename in os.listdir(folder_name + "/" + season):
            dat = np.load(folder_name + "/" + season + "/" + filename)
            total_images += 1
            for x_index in range(0, len(dat)):
                for y_index in range(0, len(dat[x_index])):
                    all_pixels.append(np.divide(dat[x_index][y_index][0:3], 10000))
            if (total_images % batch_size == 0):
                kmeans = kmeans.partial_fit(all_pixels)
                all_pixels = []
                print("Done with batch")

    if (total_images % batch_size != 0):
        kmeans = kmeans.partial_fit(all_pixels)
        print("*Done with all batches*")

    return kmeans


def predict(vocabulary, file_name, model, nclusters=20):
    NUM_CLASSES = 100
    dat = np.load(file_name)
    image_pixels = []
    for x_index in range(0, len(dat)):
        for y_index in range(0, len(dat[x_index])):
            image_pixels.append(np.divide(dat[x_index][y_index][0:3], 10000))

    kmeans = MiniBatchKMeans(n_clusters=nclusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_predictions = kmeans.fit_predict(image_pixels)
    cluster_histograms = []

    for i in range(0, nclusters):
        histogram = [0] * vocabulary.n_clusters
        cluster_histograms.append(histogram)
    pixel_no = 0
    for x_index in range(0, len(dat)):
        for y_index in range(0, len(dat[x_index])):
            cluster_point = vocabulary.predict(np.array([np.divide(dat[x_index][y_index][0:3], 10000)]))[0]
            cluster_histograms[cluster_predictions[pixel_no]][cluster_point] += 1
            pixel_no += 1

    landcover = np.zeros((256, 256), dtype=float)
    histo_classes = []

    for i in range(0, nclusters):
        input_hist = []
        normalized_histogram = np.divide(cluster_histograms[i], sum(cluster_histograms[i]))
        input_hist.append(normalized_histogram)
        input_hist = np.array(input_hist)
        predicted_class = model.predict(input_hist)
        histo_classes.append(predicted_class)

    pixel_no = 0
    for x_index in range(0, len(landcover)):
        for y_index in range(0, len(landcover[x_index])):
            landcover[x_index][y_index] = histo_classes[cluster_predictions[pixel_no]]
            pixel_no += 1

    return landcover


def predict_highlc(filename):
    with open('model_vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)

    with open('HISTO-CLASS_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    lc = predict(vocab, filename, clf)
    return lc


data_dir = "SEN12MS/processed/"
validation_states = ['ROIs13_high_roshan_s2']
validation_patches = []
for state in validation_states:
    fn = os.path.join(data_dir, '%s_val_patches/' % state)
    # print(fn)
    with os.scandir(fn) as entries:
        for entry in entries:
            # print(entry.name)
            with os.scandir(fn + entry.name) as folders:
                for folder in folders:
                    # print(folder.name)
                    with os.scandir(fn + entry.name + '/' + folder.name) as files:
                        for file in files:
                            # print(file.name)
                            validation_patches.append(
                                (os.path.join(fn + entry.name + '/' + folder.name + '/', file.name), state))
# print(validation_patches)
print("Loaded %d validation patches" % len(validation_patches))


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


iou = 0
f1 = 0

for i, namee in enumerate(validation_patches):
    y = validation_patches[i][0]
    outp = np.expand_dims(predict_highlc(y),axis=-1)
    outp = np.expand_dims(keras.utils.to_categorical(outp, 11),axis=0)
    data = np.load(validation_patches[i][0])
    tar_image = data[:, :, 5]
    tar_image = np.expand_dims(keras.utils.to_categorical(tar_image, 11),axis=0)
    iou = iou + K.get_value(iou_coef(tar_image, outp))
    f1 = f1 + K.get_value(dice_coef(tar_image, outp))
    print("Done with ", str(namee))

a = f1 / len(validation_patches)
print("The f1 score is ", a)
b = iou / len(validation_patches)
print("The iou score is ", b)