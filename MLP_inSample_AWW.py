# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:03:59 2017

@author: Kevin
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV

dataPath = 'AlanWalksWales/'
# Comment out one of the two data names to change portion of data to use
dataName = 'AWW_rest'
#dataName = 'AWW_walk'
nJobs = 12  # Number of cores to use

# Load feature matrices, labels, and groups (denoting which labeled time
# segment each row of the feature matrix comes from)
featuresAll = np.loadtxt(dataPath+dataName+'_all.csv',delimiter=',')
featuresAcc = np.loadtxt(dataPath+dataName+'_acc.csv',delimiter=',')
featuresEda = np.loadtxt(dataPath+dataName+'_eda.csv',delimiter=',')
labels = np.loadtxt(dataPath+dataName+'_label.csv')
groups = np.loadtxt(dataPath+dataName+'_groups.csv')

# Leave-one-group-out cross-validation
cv = LeaveOneGroupOut()

# Parameter tuning by grid search
solver='lbfgs'
activation='relu'
regParam = 10.0**np.arange(-3,5)

# Comment out one of the choices below (either 1 or 2 hidden layers)

# 1 hidden layer
hiddenLayerSizes = 2**np.arange(0,8)
"""
# 2 hidden layers
hidden1,hidden2 = np.meshgrid(2**np.arange(0,8),2**np.arange(0,8))
hiddenLayerSizes = np.reshape(np.stack([hidden1,hidden2]),
                                       (2,np.size(hidden1))).T.tolist()
"""
parameters = {'alpha': regParam,
              'hidden_layer_sizes': hiddenLayerSizes}
              
gsAll = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
gsAll.fit(featuresAll,labels,groups)
bestAlphaAll = gsAll.best_params_['alpha']
bestHiddenSizesAll = gsAll.best_params_['hidden_layer_sizes']

gsAcc = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
gsAcc.fit(featuresAcc,labels,groups)
bestAlphaAcc = gsAcc.best_params_['alpha']
bestHiddenSizesAcc = gsAcc.best_params_['hidden_layer_sizes']

gsEda = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
gsEda.fit(featuresEda,labels,groups)
bestAlphaEda = gsEda.best_params_['alpha']
bestHiddenSizesEda = gsEda.best_params_['hidden_layer_sizes']

predAll = np.zeros(np.shape(labels))
predAcc = np.zeros(np.shape(labels))
predEda = np.zeros(np.shape(labels))

for train, test in cv.split(featuresAll,labels,groups):
    mlpAll = MLPClassifier(hidden_layer_sizes=bestHiddenSizesAll,
                           solver=solver,alpha=bestAlphaAll)
    mlpAll.fit(featuresAll[train,:],labels[train])
    predAll[test] = mlpAll.predict_proba(featuresAll[test,:])[:,1]
    
    mlpAcc = MLPClassifier(hidden_layer_sizes=bestHiddenSizesAcc,
                           solver=solver,alpha=bestAlphaAcc)
    mlpAcc.fit(featuresAcc[train,:],labels[train])
    predAcc[test] = mlpAcc.predict_proba(featuresAcc[test,:])[:,1]

    mlpEda = MLPClassifier(hidden_layer_sizes=bestHiddenSizesEda,
                           solver=solver,alpha=bestAlphaEda)
    mlpEda.fit(featuresEda[train,:],labels[train])
    predEda[test] = mlpEda.predict_proba(featuresEda[test,:])[:,1]

# Save the scores for further analysis
#np.save('MLPpredAllScores_rest',predAll)
#np.save('MLPpredAccScores_rest',predAcc)
#np.save('MLPpredEdaScores_rest',predEda)

print('MLP AUC ALL: %f (%s)' % (roc_auc_score(labels,predAll),gsAll.best_params_))
print('MLP AUC ACC: %f (%s)' % (roc_auc_score(labels,predAcc),gsAcc.best_params_))
print('MLP AUC EDA: %f (%s)' % (roc_auc_score(labels,predEda),gsEda.best_params_))
