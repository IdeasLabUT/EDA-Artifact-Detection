# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:36:35 2017

@author: yzhang17
"""

import numpy as np
from statistics import mode
from scipy import stats
import pywt

dataPath_AWW = 'AlanWalksWales/Raw/'

dataFiles = (dataPath_AWW+'2013_04_24_1hour_walk&chat&drink.csv',dataPath_AWW+ '2013_05_06 - 1hour_walking.csv',dataPath_AWW+ '2013_05_28_1hour_walking.csv',
    dataPath_AWW+ '2013_06_04_1hour_walking.csv',dataPath_AWW+ '2013_06_10_1hour_walking.csv',dataPath_AWW+ '2013_04_24_1hour_chat&drink_Pub.csv',
    dataPath_AWW+ '2013_05_14_40mins_eating.csv',dataPath_AWW+ '2013_05_19_40mins_eat&drink.csv',dataPath_AWW+ '2013_06_03_1hour_lunch.csv',
    dataPath_AWW+ '2013_06_11_40mins_eat&readnewspaper.csv',dataPath_AWW+ '2013_07_10_1hour_Dinner.csv')

dataFiles_labels = (dataPath_AWW+ '2013_04_24_1hour_walk&chat&drink_Epochs.csv',dataPath_AWW+ '2013_05_06 - 1hour_walking_Epochs.csv',dataPath_AWW+ '2013_05_28_1hour_walking_Epochs.csv',
    dataPath_AWW+ '2013_06_04_1hour_walking_Epochs.csv',dataPath_AWW+ '2013_06_10_1hour_walking_Epochs.csv',dataPath_AWW+ '2013_04_24_1hour_chat&drink_Pub_Epochs.csv',
    dataPath_AWW+ '2013_05_14_40mins_eating_Epochs.csv',dataPath_AWW+ '2013_05_19_40mins_eat&drink_Epochs.csv',dataPath_AWW+ '2013_06_03_1hour_lunch_Epochs.csv',
    dataPath_AWW+ '2013_06_11_40mins_eat&readnewspaper_Epochs.csv',dataPath_AWW+ '2013_07_10_1hour_Dinner_Epochs.csv')

nSubjects_AWW = 11; # Number of time segments in AWW resting dataset

def statistics(data): # The function for the 4 statistics
    avg = np.mean(data) # mean
    sd = np.std(data) # standard deviation
    maxm = max(data) # maximum
    minm = min(data) # minimum
    return avg,sd,maxm,minm
    
def Derivatives(data): # Get the first and second derivatives of the data
    deriv = (data[1:-1] + data[2:])/ 2. - (data[1:-1] + data[:-2])/ 2.
    secondDeriv = data[2:] - 2*data[1:-1] + data[:-2]
    return deriv,secondDeriv

def featureMatrix(data,labels_all): # Construct the feature matrix
    length = len(labels_all)
    # Create the one label set by the majority vite
    labels = np.zeros((length,))
    for j in range(length):
        labels[j]=mode(labels_all[j,:])
    # Divide the data into 5 seconds time windows, 8Hz is the sampling rate, thus 40 data points each time window(5s).
    EDA = data[0:length*40,5].reshape(length,40)
    ACCx = data[0:length*40,0].reshape(length,40)
    ACCy = data[0:length*40,1].reshape(length,40)
    ACCz = data[0:length*40,2].reshape(length,40)
    # Get the ACC magnitude by root-mean-square
    acc = np.sqrt(np.square(data[0:length*40,0]) + np.square(data[0:length*40,1]) + np.square(data[0:length*40,2]))
    ACC = acc.reshape(length,40)
    # Construct the feature matrix, 24 EDA features, 96 ACC features, and 120 features in total. 
    features = np.zeros((length,120))
    for i in range(length):
        deriv_EDA,secondDeriv_EDA = Derivatives(EDA[i,:])
        deriv_ACC,secondDeriv_ACC = Derivatives(ACC[i,:])
        deriv_ACCx,secondDeriv_ACCx = Derivatives(ACCx[i,:])
        deriv_ACCy,secondDeriv_ACCy = Derivatives(ACCy[i,:])
        deriv_ACCz,secondDeriv_ACCz = Derivatives(ACCz[i,:])
        _, EDA_cD_3, EDA_cD_2, EDA_cD_1 = pywt.wavedec(EDA[i,:], 'Haar', level=3) #3 = 1Hz, 2 = 2Hz, 1=4Hz
        _, ACC_cD_3, ACC_cD_2, ACC_cD_1 = pywt.wavedec(ACC[i,:], 'Haar', level=3) 
        _, ACCx_cD_3, ACCx_cD_2, ACCx_cD_1 = pywt.wavedec(ACCx[i,:], 'Haar', level=3) 
        _, ACCy_cD_3, ACCy_cD_2, ACCy_cD_1 = pywt.wavedec(ACCy[i,:], 'Haar', level=3) 
        _, ACCz_cD_3, ACCz_cD_2, ACCz_cD_1 = pywt.wavedec(ACCz[i,:], 'Haar', level=3) 
        
        ### EDA features
        # EDA statistical features:
        features[i,0:4] = statistics(EDA[i,:])
        features[i,4:8] = statistics(deriv_EDA)
        features[i,8:12] = statistics(secondDeriv_EDA)
        # EDA wavelet features:
        features[i,12:16] = statistics(EDA_cD_3)
        features[i,16:20] = statistics(EDA_cD_2)
        features[i,20:24] = statistics(EDA_cD_1)
        
        ### ACC features
        ## ACC statistical features:
        # Acceleration magnitude:
        features[i,24:28] = statistics(ACC[i,:])
        features[i,28:32] = statistics(deriv_ACC)
        features[i,32:36] = statistics(secondDeriv_ACC)
        # Acceleration x-axis:
        features[i,36:40] = statistics(ACCx[i,:])
        features[i,40:44] = statistics(deriv_ACCx)
        features[i,44:48] = statistics(secondDeriv_ACCx)
        # Acceleration y-axis:
        features[i,48:52] = statistics(ACCy[i,:])
        features[i,52:56] = statistics(deriv_ACCy)
        features[i,56:60] = statistics(secondDeriv_ACCy)
        # Acceleration z-axis:
        features[i,60:64] = statistics(ACCz[i,:])
        features[i,64:68] = statistics(deriv_ACCz)
        features[i,68:72] = statistics(secondDeriv_ACCz)
        ## ACC wavelet features:
        # ACC magnitude wavelet features:
        features[i,72:76] = statistics(ACC_cD_3)
        features[i,76:80] = statistics(ACC_cD_2)
        features[i,80:84] = statistics(ACC_cD_1)
        # ACC x-axis wavelet features:
        features[i,84:88] = statistics(ACCx_cD_3)
        features[i,88:92] = statistics(ACCx_cD_2)
        features[i,92:96] = statistics(ACCx_cD_1)
        # ACC y-axis wavelet features:
        features[i,96:100] = statistics(ACCy_cD_3)
        features[i,100:104] = statistics(ACCy_cD_2)
        features[i,104:108] = statistics(ACCy_cD_1)
        # ACC z-axis wavelet features:
        features[i,108:112] = statistics(ACCz_cD_3)
        features[i,112:116] = statistics(ACCz_cD_2)
        features[i,116:120] = statistics(ACCz_cD_1)
        
    featuresAll = stats.zscore(features) # Normalize the data using z-score
    featuresAcc = featuresAll[:,24:120] # 96 ACC features
    featuresEda = featuresAll[:,0:24] #24 EDA features
    return featuresAll,featuresAcc,featuresEda,labels

# Load the data and construct the feature matrix
data_AWW   = dict()
labels_AWW = dict()
awwAll = dict()
awwAcc = dict()
awwEda = dict()
awwLabels = dict()
awwGroups = dict()  
for i in range(nSubjects_AWW):
    data_AWW[i] = np.loadtxt(dataFiles[i], delimiter=',', skiprows=8)
    labels_AWW[i] = np.loadtxt(dataFiles_labels[i], delimiter=',', skiprows=1, usecols=(3,4,5))
    labels_AWW[i][labels_AWW[i]==0]=1 # Assume the unlabeled time windows as clean
    labels_AWW[i] = labels_AWW[i]-1 # Make the labels include only 0s and 1s
    awwGroups[i] = np.ones(len(labels_AWW[i]))*i # The group number for the leave one group out cross-validation
    awwAll[i],awwAcc[i],awwEda[i],awwLabels[i] = featureMatrix(data_AWW[i],labels_AWW[i])

# Convert the dictionary to arrays  
awwAll_walk = np.concatenate([awwAll[x] for x in range(5)], 0)
awwAll_rest = np.concatenate([awwAll[x] for x in range(5,11)], 0)

awwAcc_walk = np.concatenate([awwAcc[x] for x in range(5)], 0)
awwAcc_rest = np.concatenate([awwAcc[x] for x in range(5,11)], 0)

awwEda_walk = np.concatenate([awwEda[x] for x in range(5)], 0)
awwEda_rest = np.concatenate([awwEda[x] for x in range(5,11)], 0)

awwLabels_walk = np.concatenate([awwLabels[x] for x in range(5)], 0)
awwLabels_rest = np.concatenate([awwLabels[x] for x in range(5,11)], 0)

awwGroups_walk = np.concatenate([awwGroups[x] for x in range(5)], 0)
awwGroups_rest = np.concatenate([awwGroups[x] for x in range(5,11)], 0)