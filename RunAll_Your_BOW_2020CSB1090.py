
import tensorflow as tf 
#Loading dataset using tensorflow
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Importing Some basic libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import cv2
import skimage.io as io
import joblib
import pylab as pl
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import precision_score, recall_score

class My_Kmeancluster:
    def __init__(self, M, number,number_of_iterations):
        self.K = number 
        self.number_of_iterations = number_of_iterations 
        self.num_examples, self.num_fea = M.shape 
        self.pfig = True 
        
    def init_rancen(self, M): 
        centro = np.zeros((self.K, self.num_fea))  
        for k in range(self.K): 
            centroid = M[np.random.choice(range(self.num_examples))] 
            centro[k] = centroid
        return centro 
    
    def cre_clus(self, M, centro):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(M):
            clcen = np.argmin(
                np.sqrt(np.sum((point-centro)**2, axis=1))
            )
            clusters[clcen].append(point_idx)
        return clusters 
    
    def cal_ne_cen(self, clusters, M):
        centro = np.zeros((self.K, self.num_fea)) 
        for idx, cluster in enumerate(clusters):
            if len(cluster)!=0:
              new_centroid = np.mean(M[cluster], axis=0) 
              centro[idx] = new_centroid
        return centro
    
    def pre_clus(self, clusters, M):
        y_pred = np.zeros(self.num_examples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
    
    def fit(self, M):
        centro = self.init_rancen(M) 
        for _ in range(self.number_of_iterations):
            clusters = self.cre_clus(M, centro) 
            previous_centroids = centro
            centro = self.cal_ne_cen(clusters, M) 
            diff = centro - previous_centroids 
            if not diff.any():
                break
        y_pred = self.pre_clus(clusters, M) 
        return centro


Array_for_pitures = []
Classes_for_pitures=[]

def CreateDictionary():
    read_from_csv = pd.read_csv('dataset/fashion-mnist_train.csv')
    global Array_for_pitures
    global Classes_for_pitures
    jkl = read_from_csv['label']
    for x in jkl:
        Classes_for_pitures.append(x)
    d = read_from_csv.drop("label",axis=1)
    for i in range(0,d.shape[0]):
        grid_data = d.iloc[i].to_numpy().reshape(28,28)
        Array_for_pitures.append(grid_data.astype(np.uint8))
    print ("Dictionary created")
    return
    
    
def computeHistogram():
    global Array_for_pitures,Classes_for_pitures
    key =[]
    des = []
    sift_descrip = cv2.xfeatures2d.SIFT_create(100)
    image_classes2=[]
    for i in range(len(Array_for_pitures)):
        keypoints, descriptor = sift_descrip.detectAndCompute(Array_for_pitures[i], None)
        if descriptor is None:
            continue
        image_classes2.append(Classes_for_pitures[i])
        key.append(keypoints)
        des.append(descriptor)


    descriptors=np.vstack(des)
    descriptors_float=descriptors.astype(float)
    k = 225  #k means with k clusters
    mean_k = My_Kmeancluster(descriptors_float, k, 30)
    voc = mean_k.fit(descriptors_float) 
    
    im_features = np.zeros((len(image_classes2), k), "float32")
    for i in range(len(image_classes2)):
        words_arr, distance_Arr = vq(des[i],voc)
        for w in words_arr:
            im_features[i][w] += 1

    from sklearn.preprocessing import StandardScaler
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)
    
    from sklearn.svm import LinearSVC
    clf = LinearSVC(max_iter=1000) 
    clf.fit(im_features, np.array(image_classes2))
    print ("Histogram  Computed")
    training_names=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    joblib.dump((clf, training_names, stdSlr, k, voc), "bovw.pkl", compress=3) 


def matchHistogram():
    print ("Histogram  Matching Going on")
    clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")
    reading = pd.read_csv('dataset/fashion-mnist_test.csv')
    Classes_for_pitures=[]
    image_label = reading['label']
    for x in image_label:
        Classes_for_pitures.append(x)
    d = reading.drop("label",axis=1)
    Array_for_pitures = []    
    for i in range(0,d.shape[0]):
        grid_data = d.iloc[i].to_numpy().reshape(28,28)
        Array_for_pitures.append(grid_data.astype(np.uint8))
        
    key =[]
    des = []
    sift_descrip = cv2.xfeatures2d.SIFT_create(100)
    image_classes2=[]
    for i in range(len(Array_for_pitures)):
        keypoints, descriptor = sift_descrip.detectAndCompute(Array_for_pitures[i], None)
        if descriptor is None:
            continue
        image_classes2.append(Classes_for_pitures[i])
        key.append(keypoints)
        des.append(descriptor)
    
    
    im_features = np.zeros((len(image_classes2), k), "float32")
    for i in range(len(image_classes2)):
        words_arr, distance_Arr = vq(des[i],voc)
        for w in words_arr:
            im_features[i][w] += 1
            
    im_features = stdSlr.transform(im_features)
    
    correct_class =  [classes_names[i] for i in image_classes2] 
    predict_class =  [classes_names[i] for i in clf.predict(im_features)]
    
    count=0
    tot_count=0
    for i in range(len(correct_class)):
        tot_count+=1
        if(correct_class[i]==predict_class[i]):
            count+=1
    
    print ("Histograms  Matched")
    print ("True_class ="  + str(correct_class))
    print ("Prediction ="  + str(predict_class))
    
    
    
    print("Total Number of Hits Are:",count)
    accuracy = accuracy_score(correct_class, predict_class)
    print ("Your overall classification accuracy is = ", accuracy)
    print ("Precison is" , precision_score(correct_class, predict_class,average='micro'))
    print ("Recall is" , recall_score(correct_class, predict_class ,average='micro'))
    cm = confusion_matrix(correct_class, predict_class)
    print (cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Accuracy Of each class is:")
    cm.diagonal()
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()
    

CreateDictionary()
computeHistogram()
matchHistogram()



