# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:58:22 2017

@author: Kevin
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV

utdPath = 'UTDallas/'
utdName = 'UTD'
awwPath = 'AlanWalksWales/'
awwName1 = 'AWW_rest'
awwName2 = 'AWW_walk'
nJobs = 12  # Number of cores to use

# Load UTD data and identify rows to retain when doing cross-validation
# (some subjects do not have any motion artifacts and need to be excluded 
# from the leave-one-subject-out cross-validation)
utdAll = np.loadtxt(utdPath+utdName+'_all.csv',delimiter=',')
utdAcc = np.loadtxt(utdPath+utdName+'_acc.csv',delimiter=',')
utdEda = np.loadtxt(utdPath+utdName+'_eda.csv',delimiter=',')
utdLabels = np.loadtxt(utdPath+utdName+'_label.csv')
utdGroups = np.loadtxt(utdPath+utdName+'_groups.csv')
includeRowsTrain = np.logical_and(
    np.logical_and(np.where(utdGroups!=5,True,False),
    np.where(utdGroups!=17,True,False)),np.where(utdGroups!=18,True,False))

# Load AWW data and merge together resting and walking data. Need to add 6 to
# group numbers for AWW walking so that group numbers don't overlap
awwAll = np.r_[np.loadtxt(awwPath+awwName1+'_all.csv',delimiter=','),
               np.loadtxt(awwPath+awwName2+'_all.csv',delimiter=',')]
awwAcc = np.r_[np.loadtxt(awwPath+awwName1+'_acc.csv',delimiter=','),
               np.loadtxt(awwPath+awwName2+'_acc.csv',delimiter=',')]
awwEda = np.r_[np.loadtxt(awwPath+awwName1+'_eda.csv',delimiter=','),
               np.loadtxt(awwPath+awwName2+'_eda.csv',delimiter=',')]
awwLabels = np.r_[np.loadtxt(awwPath+awwName1+'_label.csv'),
                  np.loadtxt(awwPath+awwName2+'_label.csv')]
awwGroups = np.r_[np.loadtxt(awwPath+awwName1+'_groups.csv'),
                  np.loadtxt(awwPath+awwName2+'_groups.csv')]

# leave-one-subject-out cross-validation
utdCv = LeaveOneGroupOut()
awwCv = LeaveOneGroupOut()

solver='lbfgs'
activation='relu'
regParam = 10.0**-np.arange(-3,7)
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

# Perform grid search cross-validation on UTD data to identify best parameters
utdGsAll = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                        parameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
utdGsAll.fit(utdAll[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
utdBestRegParamAll = utdGsAll.best_params_['alpha']
utdBestHiddenSizesAll = utdGsAll.best_params_['hidden_layer_sizes']

utdGsAcc = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                        parameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
utdGsAcc.fit(utdAcc[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
utdBestRegParamAcc = utdGsAcc.best_params_['alpha']
utdBestHiddenSizesAcc = utdGsAcc.best_params_['hidden_layer_sizes']

utdGsEda = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                        parameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
utdGsEda.fit(utdEda[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
utdBestRegParamEda = utdGsEda.best_params_['alpha']
utdBestHiddenSizesEda = utdGsEda.best_params_['hidden_layer_sizes']

# Fit classifier with best parameters from grid search CV to entire UTD data
# (including excluded test subjects) and test on AWW data
utdLrAll = MLPClassifier(solver=solver,activation=activation,
                         alpha=utdBestRegParamAll,
                         hidden_layer_sizes=utdBestHiddenSizesAll)
utdLrAll.fit(utdAll,utdLabels)
awwPredAll = utdLrAll.predict_proba(awwAll)[:,1]

utdLrAcc = MLPClassifier(solver=solver,activation=activation,
                         alpha=utdBestRegParamAcc,
                         hidden_layer_sizes=utdBestHiddenSizesAcc)
utdLrAcc.fit(utdAcc,utdLabels)
awwPredAcc = utdLrAcc.predict_proba(awwAcc)[:,1]

utdLrEda = MLPClassifier(solver=solver,activation=activation,
                         alpha=utdBestRegParamEda,
                         hidden_layer_sizes=utdBestHiddenSizesEda)
utdLrEda.fit(utdEda,utdLabels)
awwPredEda = utdLrEda.predict_proba(awwEda)[:,1]

# Save the scores for further analysis
#np.save('awwPredAll_MLP',awwPredAll)
#np.save('awwPredAcc_MLP',awwPredAcc)
#np.save('awwPredEda_MLP',awwPredEda)

print('Train on UTD, test on AWW')
print('AUC ALL: %f (%s)' % (roc_auc_score(awwLabels,awwPredAll),
                            utdGsAll.best_params_))
print('AUC ACC: %f (%s)' % (roc_auc_score(awwLabels,awwPredAcc),
                            utdGsAcc.best_params_))
print('AUC EDA: %f (%s)' % (roc_auc_score(awwLabels,awwPredEda),
                            utdGsEda.best_params_))

# Perform grid search cross-validation on AWW data to identify best parameters
# and re-fit entire data set with best parameters. Test on UTD data.
awwGsAll = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                        parameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=False,
                        verbose=1)
awwGsAll.fit(awwAll,awwLabels,awwGroups)
awwBestRegParamAll = awwGsAll.best_params_['alpha']
awwBestHiddenSizesAll = awwGsAll.best_params_['hidden_layer_sizes']

awwGsAcc = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                        parameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=False,
                        verbose=1)
awwGsAcc.fit(awwAcc,awwLabels,awwGroups)
awwBestRegParamAcc = awwGsAcc.best_params_['alpha']
awwBestHiddenSizesAcc = awwGsAcc.best_params_['hidden_layer_sizes']

awwGsEda = GridSearchCV(MLPClassifier(solver=solver,activation=activation),
                        parameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=False,
                        verbose=1)
awwGsEda.fit(awwEda,awwLabels,awwGroups)
awwBestRegParamEda = awwGsEda.best_params_['alpha']
awwBestHiddenSizesEda = awwGsEda.best_params_['hidden_layer_sizes']

# Fit classifier with best parameters from grid search CV to entire UTD data
# (including excluded test subjects) and test on AWW data
utdLrAll = MLPClassifier(solver=solver,activation=activation,
                         alpha=awwBestRegParamAll,
                         hidden_layer_sizes=awwBestHiddenSizesAll)
utdLrAll.fit(awwAll,awwLabels)
utdPredAll = utdLrAll.predict_proba(utdAll)[:,1]

utdLrAcc = MLPClassifier(solver=solver,activation=activation,
                         alpha=awwBestRegParamAcc,
                         hidden_layer_sizes=awwBestHiddenSizesAcc)
utdLrAcc.fit(awwAcc,awwLabels)
utdPredAcc = utdLrAcc.predict_proba(utdAcc)[:,1]

utdLrEda = MLPClassifier(solver=solver,activation=activation,
                         alpha=awwBestRegParamEda,
                         hidden_layer_sizes=awwBestHiddenSizesEda)
utdLrEda.fit(awwEda,awwLabels)
utdPredEda = utdLrEda.predict_proba(utdEda)[:,1]

# Save the scores for further analysis
#np.save('utdPredAll_MLP',utdPredAll)
#np.save('utdPredAcc_MLP',utdPredAcc)
#np.save('utdPredEda_MLP',utdPredEda)

print('Train on AWW, test on UTD')
print('AUC ALL: %f (%s)' % (roc_auc_score(utdLabels,utdPredAll),
                            awwGsAll.best_params_))
print('AUC ACC: %f (%s)' % (roc_auc_score(utdLabels,utdPredAcc),
                            awwGsAcc.best_params_))
print('AUC EDA: %f (%s)' % (roc_auc_score(utdLabels,utdPredEda),
                            awwGsEda.best_params_))
