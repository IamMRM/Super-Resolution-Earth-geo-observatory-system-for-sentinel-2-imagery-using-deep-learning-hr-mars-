import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MiniBatchKMeans
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

def makeVocabulary(training_patches, batch_size = 50):
    all_pixels = []
    total_images = 0
    kmeans = MiniBatchKMeans(n_clusters = 80, random_state = 0, max_iter=300, n_init=10)
    for folder_name in training_patches:
        for season in os.listdir(folder_name + "/"):
            for filename in os.listdir(folder_name + "/" + season):
                dat = np.load(folder_name+"/"+season+"/"+filename)
                total_images += 1
                for x_index in range(0, len(dat)):
                    for y_index in range(0, len(dat[x_index])):
                        all_pixels.append(np.divide(dat[x_index][y_index][0:3],10000))
                if(total_images % batch_size == 0):
                    kmeans = kmeans.partial_fit(all_pixels)
                    all_pixels = []
                    print("Done with batch")
                
    if(total_images % batch_size != 0):
        kmeans = kmeans.partial_fit(all_pixels)
    
    print("**Done with all batches**")
    
    return kmeans



def makeDataset(vocabulary, training_patches, nclusters = 20):
    NUM_CLASSES=100
    cluster_dataset = {"histograms": [], "labels": []}

    for folder_name in training_patches:
        for season in os.listdir(folder_name + "/"):
            for filename in os.listdir(folder_name+"/"+season):
                dat = np.load(folder_name+"/"+season+"/"+filename)
                image_pixels = []
                for x_index in range(0, len(dat)):
                     for y_index in range(0, len(dat[x_index])):
                            image_pixels.append(np.divide(dat[x_index][y_index][0:3],10000))
                image_pixels=np.array(image_pixels)
                kmeans = MiniBatchKMeans(n_clusters=nclusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
                cluster_predictions = kmeans.fit_predict(image_pixels)

                cluster_histograms = []
                cluster_y = []

                for i in range(0, nclusters):
                    cluster_y.append([0]*NUM_CLASSES)
                    histogram = [0]*vocabulary.n_clusters
                    cluster_histograms.append(histogram)

                pixel_no = 0
                for x_index in range(0, len(dat)):
                     for y_index in range(0, len(dat[x_index])):
                         cluster_point = vocabulary.predict(np.array([np.divide(dat[x_index][y_index][0:3],10000)]))[0]
                         cluster_histograms[cluster_predictions[pixel_no]][cluster_point] += 1
                         cluster_y[cluster_predictions[pixel_no]][int(dat[x_index][y_index][5])] += 1
                         pixel_no += 1

                for i in range(0, nclusters):
                    cluster_dataset['labels'].append(cluster_y[i].index(max(cluster_y[i])))
                    normalized_histogram = np.divide(cluster_histograms[i], sum(cluster_histograms[i]))
                    cluster_dataset['histograms'].append(normalized_histogram)
            
    return cluster_dataset
            
                    
training_patches=["SEN12MS/processed/ROIs13_high_roshan_s2_train_patches/ROIs13_high_roshan_s2"]
#training_patches = ["../SEN12MS/processed/ROIs1868_summer_s2_train_patches/ROIs1868_summer","../SEN12MS/processed/ROIs1970_fall_s2_train_patches/ROIs1970_fall","../SEN12MS/processed/ROIs1158_spring_s2_train_patches/ROIs1158_spring","../SEN12MS/processed/ROI2017_winter_s2_train_patches/ROI2017_winter"]

print("Starting program")
vocab = makeVocabulary(training_patches)

print("Starting dataset")
dataset = makeDataset(vocab, training_patches)

print("Starting random forest classifier")
# training on the dataset
clf=RandomForestClassifier(n_estimators=1000,n_jobs=-1)
clf.fit(dataset['histograms'], dataset['labels'])

with open('HISTO-CLASS_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
    
with open('model_vocabulary.pkl', 'wb') as f:
    pickle.dump(vocab, f)

np.save("train_set",dataset)





















