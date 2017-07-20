# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:59:26 2017

@author: yzhang17
"""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier,IsolationForest

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

## Logistic Regression
LRparameters = {'C': 10.0**-np.arange(-3,7)}
# Perform grid search cross-validation on UTD data to identify best parameters
LRutdGsAll = GridSearchCV(LogisticRegression(),
                        LRparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
LRutdGsAll.fit(utdAll[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
LRutdBestC_All = LRutdGsAll.best_params_['C']

LRutdGsAcc = GridSearchCV(LogisticRegression(),
                        LRparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
LRutdGsAcc.fit(utdAcc[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
LRutdBestC_Acc = LRutdGsAcc.best_params_['C']

LRutdGsEda = GridSearchCV(LogisticRegression(),
                        LRparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
LRutdGsEda.fit(utdEda[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
LRutdBestC_Eda = LRutdGsEda.best_params_['C']

# Fit classifier with best parameters from grid search CV to entire UTD data
# (including excluded test subjects) and test on AWW data
utdLrAll = LogisticRegression(C=LRutdBestC_All)
utdLrAll.fit(utdAll,utdLabels)
LRawwPredAll = utdLrAll.predict_proba(awwAll)[:,1]

utdLrAcc = LogisticRegression(C=LRutdBestC_Acc)
utdLrAcc.fit(utdAcc,utdLabels)
LRawwPredAcc = utdLrAcc.predict_proba(awwAcc)[:,1]

utdLrEda = LogisticRegression(C=LRutdBestC_Eda)
utdLrEda.fit(utdEda,utdLabels)
LRawwPredEda = utdLrEda.predict_proba(awwEda)[:,1]

# Save the scores for further analysis
#np.save('awwPredAll_LR',LRawwPredAll)
#np.save('awwPredAcc_LR',LRawwPredAcc)
#np.save('awwPredEda_LR',LRawwPredEda)

print('Train on UTD, test on AWW')
print('LR AUC ALL: %f (%s)' % (roc_auc_score(awwLabels,LRawwPredAll),
                            LRutdGsAll.best_params_))
print('LR AUC ACC: %f (%s)' % (roc_auc_score(awwLabels,LRawwPredAcc),
                            LRutdGsAcc.best_params_))
print('LR AUC EDA: %f (%s)' % (roc_auc_score(awwLabels,LRawwPredEda),
                            LRutdGsEda.best_params_))


# Perform grid search cross-validation on AWW data to identify best parameters
# and re-fit entire data set with best parameters. Test on UTD data.
LRawwGsAll = GridSearchCV(LogisticRegression(),
                        LRparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
LRawwGsAll.fit(awwAll,awwLabels,awwGroups)
LRawwBestC_All = LRawwGsAll.best_params_['C']

LRawwGsAcc = GridSearchCV(LogisticRegression(),
                        LRparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
LRawwGsAcc.fit(awwAcc,awwLabels,awwGroups)
LRawwBestC_Acc = LRawwGsAcc.best_params_['C']

LRawwGsEda = GridSearchCV(LogisticRegression(),
                        LRparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
LRawwGsEda.fit(awwEda,awwLabels,awwGroups)
LRawwBestC_Eda = LRawwGsEda.best_params_['C']

awwLrAll = LogisticRegression(C=LRawwBestC_All)
awwLrAll.fit(awwAll,awwLabels)
LRutdPredAll = awwLrAll.predict_proba(utdAll)[:,1]

awwLrAcc = LogisticRegression(C=LRawwBestC_Acc)
awwLrAcc.fit(awwAcc,awwLabels)
LRutdPredAcc = awwLrAcc.predict_proba(utdAcc)[:,1]

awwLrEda = LogisticRegression(C=LRawwBestC_Eda)
awwLrEda.fit(awwEda,awwLabels)
LRutdPredEda = awwLrEda.predict_proba(utdEda)[:,1]

# Save the scores for further analysis
#np.save('utdPredAll_LR',LRutdPredAll)
#np.save('utdPredAcc_LR',LRutdPredAcc)
#np.save('utdPredEda_LR',LRutdPredEda)

print('Train on AWW, test on UTD')
print('LR AUC ALL: %f (%s)' % (roc_auc_score(utdLabels,LRutdPredAll),
                            LRawwGsAll.best_params_))
print('LR AUC ACC: %f (%s)' % (roc_auc_score(utdLabels,LRutdPredAcc),
                            LRawwGsAcc.best_params_))
print('LR AUC EDA: %f (%s)' % (roc_auc_score(utdLabels,LRutdPredEda),
                            LRawwGsEda.best_params_))


## SVM
SVMparameters = {'gamma': 10. ** np.arange(-8, 1), 'C': 10. ** np.arange(-3, 5)}

SVMutdGsAll = GridSearchCV(svm.SVC(),
                        SVMparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
SVMutdGsAll.fit(utdAll[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
SVMutdBestC_All = SVMutdGsAll.best_params_['C']
SVMutdbestgamma_All = SVMutdGsAll.best_params_['gamma']

SVMutdGsAcc = GridSearchCV(svm.SVC(),
                        SVMparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
SVMutdGsAcc.fit(utdAcc[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
SVMutdBestC_Acc = SVMutdGsAcc.best_params_['C']
SVMutdbestgamma_Acc = SVMutdGsAcc.best_params_['gamma']

SVMutdGsEda = GridSearchCV(svm.SVC(),
                        SVMparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
SVMutdGsEda.fit(utdEda[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
SVMutdBestC_Eda = SVMutdGsEda.best_params_['C']
SVMutdbestgamma_Eda = SVMutdGsEda.best_params_['gamma']

# Fit classifier with best parameters from grid search CV to entire UTD data
# (including excluded test subjects) and test on AWW data
SVMutdSvmAll = svm.SVC(gamma=SVMutdbestgamma_All, C=SVMutdBestC_All, kernel='rbf',probability=True)
SVMutdSvmAll.fit(utdAll,utdLabels)
SVMawwPredAll = SVMutdSvmAll.predict_proba(awwAll)[:,1]

SVMutdSvmAcc = svm.SVC(gamma=SVMutdbestgamma_Acc, C=SVMutdBestC_Acc, kernel='rbf',probability=True)
SVMutdSvmAcc.fit(utdAcc,utdLabels)
SVMawwPredAcc = SVMutdSvmAcc.predict_proba(awwAcc)[:,1]

SVMutdSvmEda = svm.SVC(gamma=SVMutdbestgamma_Eda, C=SVMutdBestC_Eda, kernel='rbf',probability=True)
SVMutdSvmEda.fit(utdEda,utdLabels)
SVMawwPredEda = SVMutdSvmEda.predict_proba(awwEda)[:,1]

# Save the scores for further analysis
#np.save('awwPredAll_SVM',SVMawwPredAll)
#np.save('awwPredAcc_SVM',SVMawwPredAcc)
#np.save('awwPredEda_SVM',SVMawwPredEda)

print('Train on UTD, test on AWW')
print('SVM AUC ALL: %f (%s)' % (roc_auc_score(awwLabels,SVMawwPredAll),
                            SVMutdGsAll.best_params_))
print('SVM AUC ACC: %f (%s)' % (roc_auc_score(awwLabels,SVMawwPredAcc),
                            SVMutdGsAcc.best_params_))
print('SVM AUC EDA: %f (%s)' % (roc_auc_score(awwLabels,SVMawwPredEda),
                            SVMutdGsEda.best_params_))

# Perform grid search cross-validation on AWW data to identify best parameters
# and re-fit entire data set with best parameters. Test on UTD data.
SVMawwGsAll = GridSearchCV(svm.SVC(),
                        SVMparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
SVMawwGsAll.fit(awwAll,awwLabels,awwGroups)
SVMawwBestC_All = SVMawwGsAll.best_params_['C']
SVMawwbestgamma_All = SVMawwGsAll.best_params_['gamma']

SVMawwGsAcc = GridSearchCV(svm.SVC(),
                        SVMparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
SVMawwGsAcc.fit(awwAcc,awwLabels,awwGroups)
SVMawwBestC_Acc = SVMawwGsAcc.best_params_['C']
SVMawwbestgamma_Acc = SVMawwGsAcc.best_params_['gamma']

SVMawwGsEda = GridSearchCV(svm.SVC(),
                        SVMparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
SVMawwGsEda.fit(awwEda,awwLabels,awwGroups)
SVMawwBestC_Eda = SVMawwGsEda.best_params_['C']
SVMawwbestgamma_Eda = SVMawwGsEda.best_params_['gamma']

awwSvmAll = svm.SVC(gamma=SVMawwbestgamma_All, C=SVMawwBestC_All, kernel='rbf',probability=True)
awwSvmAll.fit(awwAll,awwLabels)
SVMutdPredAll = awwSvmAll.predict_proba(utdAll)[:,1]

awwSvmAcc = svm.SVC(gamma=SVMawwbestgamma_Acc, C=SVMawwBestC_Acc, kernel='rbf',probability=True)
awwSvmAcc.fit(awwAcc,awwLabels)
SVMutdPredAcc = awwSvmAcc.predict_proba(utdAcc)[:,1]

awwSvmEda = svm.SVC(gamma=SVMawwbestgamma_Eda, C=SVMawwBestC_Eda, kernel='rbf',probability=True)
awwSvmEda.fit(awwEda,awwLabels)
SVMutdPredEda = awwSvmEda.predict_proba(utdEda)[:,1]

# Save the scores for further analysis
#np.save('utdPredAll_SVM',SVMutdPredAll)
#np.save('utdPredAcc_SVM',SVMutdPredAcc)
#np.save('utdPredEda_SVM',SVMutdPredEda)

print('Train on AWW, test on UTD')
print('SVM AUC ALL: %f (%s)' % (roc_auc_score(utdLabels,SVMutdPredAll),
                            SVMawwGsAll.best_params_))
print('SVM AUC ACC: %f (%s)' % (roc_auc_score(utdLabels,SVMutdPredAcc),
                            SVMawwGsAcc.best_params_))
print('SVM AUC EDA: %f (%s)' % (roc_auc_score(utdLabels,SVMutdPredEda),
                            SVMawwGsEda.best_params_))


## kNN classification
# Perform grid search cross-validation on UTD data to identify best parameters
kNNCparameters = {'n_neighbors': 2 ** np.arange(0, 11)}

kNNCutdGsAll = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                        kNNCparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
kNNCutdGsAll.fit(utdAll[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
kNNCutdBestneighbors_All = kNNCutdGsAll.best_params_['n_neighbors']

kNNCutdGsAcc = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                        kNNCparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
kNNCutdGsAcc.fit(utdAcc[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
kNNCutdBestneighbors_Acc = kNNCutdGsAcc.best_params_['n_neighbors']

kNNCutdGsEda = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                        kNNCparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
kNNCutdGsEda.fit(utdEda[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
kNNCutdBestneighbors_Eda = kNNCutdGsEda.best_params_['n_neighbors']

# Fit classifier with best parameters from grid search CV to entire UTD data
# (including excluded test subjects) and test on AWW data
utdkNNCAll = KNeighborsClassifier(n_neighbors=kNNCutdBestneighbors_All, algorithm='auto', metric='euclidean')
utdkNNCAll.fit(utdAll,utdLabels)
kNNCawwPredAll = utdkNNCAll.predict_proba(awwAll)[:,1]

utdkNNCAcc = KNeighborsClassifier(n_neighbors=kNNCutdBestneighbors_Acc, algorithm='auto', metric='euclidean')
utdkNNCAcc.fit(utdAcc,utdLabels)
kNNCawwPredAcc = utdkNNCAcc.predict_proba(awwAcc)[:,1]

utdkNNCEda = KNeighborsClassifier(n_neighbors=kNNCutdBestneighbors_Eda, algorithm='auto', metric='euclidean')
utdkNNCEda.fit(utdEda,utdLabels)
kNNCawwPredEda = utdkNNCEda.predict_proba(awwEda)[:,1]

# Save the scores for further analysis
#np.save('awwPredAll_kNNclass',kNNCawwPredAll)
#np.save('awwPredAcc_kNNclass',kNNCawwPredAcc)
#np.save('awwPredEda_kNNclass',kNNCawwPredEda)

print('Train on UTD, test on AWW')
print('kNNclass AUC ALL: %f (%s)' % (roc_auc_score(awwLabels,kNNCawwPredAll),
                            kNNCutdGsAll.best_params_))
print('kNNclass AUC ACC: %f (%s)' % (roc_auc_score(awwLabels,kNNCawwPredAcc),
                            kNNCutdGsAcc.best_params_))
print('kNNclass AUC EDA: %f (%s)' % (roc_auc_score(awwLabels,kNNCawwPredEda),
                            kNNCutdGsEda.best_params_))

# Perform grid search cross-validation on AWW data to identify best parameters
# and re-fit entire data set with best parameters. Test on UTD data.
kNNCawwGsAll = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                        kNNCparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
kNNCawwGsAll.fit(awwAll,awwLabels,awwGroups)
kNNCawwBestneighbors_All = kNNCawwGsAll.best_params_['n_neighbors']

kNNCawwGsAcc = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                        kNNCparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
kNNCawwGsAcc.fit(awwAcc,awwLabels,awwGroups)
kNNCawwBestneighbors_Acc = kNNCawwGsAcc.best_params_['n_neighbors']

kNNCawwGsEda = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                        kNNCparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
kNNCawwGsEda.fit(awwEda,awwLabels,awwGroups)
kNNCawwBestneighbors_Eda = kNNCawwGsEda.best_params_['n_neighbors']

awwkNNCAll = KNeighborsClassifier(n_neighbors=kNNCawwBestneighbors_All, algorithm='auto', metric='euclidean')
awwkNNCAll.fit(awwAll,awwLabels)
kNNCutdPredAll = awwkNNCAll.predict_proba(utdAll)[:,1]

awwkNNCAcc = KNeighborsClassifier(n_neighbors=kNNCawwBestneighbors_Acc, algorithm='auto', metric='euclidean')
awwkNNCAcc.fit(awwAcc,awwLabels)
kNNCutdPredAcc = awwkNNCAcc.predict_proba(utdAcc)[:,1]

awwkNNCEda = KNeighborsClassifier(n_neighbors=kNNCawwBestneighbors_Eda, algorithm='auto', metric='euclidean')
awwkNNCEda.fit(awwEda,awwLabels)
kNNCutdPredEda = awwkNNCEda.predict_proba(utdEda)[:,1]

# Save the scores for further analysis
#np.save('utdPredAll_kNNclass',kNNCutdPredAll)
#np.save('utdPredAcc_kNNclass',kNNCutdPredAcc)
#np.save('utdPredEda_kNNclass',kNNCutdPredEda)

print('Train on AWW, test on UTD')
print('kNNclass AUC ALL: %f (%s)' % (roc_auc_score(utdLabels,kNNCutdPredAll),
                            kNNCawwGsAll.best_params_))
print('kNNclass AUC ACC: %f (%s)' % (roc_auc_score(utdLabels,kNNCutdPredAcc),
                            kNNCawwGsAcc.best_params_))
print('kNNclass AUC EDA: %f (%s)' % (roc_auc_score(utdLabels,kNNCutdPredEda),
                            kNNCawwGsEda.best_params_))


## Random Forest
# Perform grid search cross-validation on UTD data to identify best parameters
RFparameters = {'n_estimators': 10*np.arange(1,21)}

RFutdGsAll = GridSearchCV(RandomForestClassifier(),
                        RFparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
RFutdGsAll.fit(utdAll[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
RFutdBestNumTrees_All = RFutdGsAll.best_params_['n_estimators']

RFutdGsAcc = GridSearchCV(RandomForestClassifier(),
                        RFparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
RFutdGsAcc.fit(utdAcc[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
RFutdBestNumTrees_Acc = RFutdGsAcc.best_params_['n_estimators']

RFutdGsEda = GridSearchCV(RandomForestClassifier(),
                        RFparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
RFutdGsEda.fit(utdEda[includeRowsTrain,:],utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
RFutdBestNumTrees_Eda = RFutdGsEda.best_params_['n_estimators']

# Fit classifier with best parameters from grid search CV to entire UTD data
# (including excluded test subjects) and test on AWW data
utdRFAll = RandomForestClassifier(n_estimators=RFutdBestNumTrees_All)
utdRFAll.fit(utdAll,utdLabels)
RFawwPredAll = utdRFAll.predict_proba(awwAll)[:,1]

utdRFAcc = RandomForestClassifier(n_estimators=RFutdBestNumTrees_Acc)
utdRFAcc.fit(utdAcc,utdLabels)
RFawwPredAcc = utdRFAcc.predict_proba(awwAcc)[:,1]

utdRFEda = RandomForestClassifier(n_estimators=RFutdBestNumTrees_Eda)
utdRFEda.fit(utdEda,utdLabels)
RFawwPredEda = utdRFEda.predict_proba(awwEda)[:,1]

# Save the scores for further analysis
#np.save('awwPredAll_RF',RFawwPredAll)
#np.save('awwPredAcc_RF',RFawwPredAcc)
#np.save('awwPredEda_RF',RFawwPredEda)

print('Train on UTD, test on AWW')
print('RF AUC ALL: %f (%s)' % (roc_auc_score(awwLabels,RFawwPredAll),
                            RFutdGsAll.best_params_))
print('RF AUC ACC: %f (%s)' % (roc_auc_score(awwLabels,RFawwPredAcc),
                            RFutdGsAcc.best_params_))
print('RF AUC EDA: %f (%s)' % (roc_auc_score(awwLabels,RFawwPredEda),
                            RFutdGsEda.best_params_))

# Perform grid search cross-validation on AWW data to identify best parameters
# and re-fit entire data set with best parameters. Test on UTD data.
RFawwGsAll = GridSearchCV(RandomForestClassifier(),
                        RFparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
RFawwGsAll.fit(awwAll,awwLabels,awwGroups)
RFawwBestNumTrees_All = RFawwGsAll.best_params_['n_estimators']

RFawwGsAcc = GridSearchCV(RandomForestClassifier(),
                        RFparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
RFawwGsAcc.fit(awwAcc,awwLabels,awwGroups)
RFawwBestNumTrees_Acc = RFawwGsAcc.best_params_['n_estimators']

RFawwGsEda = GridSearchCV(RandomForestClassifier(),
                        RFparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
RFawwGsEda.fit(awwEda,awwLabels,awwGroups)
RFawwBestNumTrees_Eda = RFawwGsEda.best_params_['n_estimators']

awwRFAll = RandomForestClassifier(n_estimators=RFawwBestNumTrees_All)
awwRFAll.fit(awwAll,awwLabels)
RFutdPredAll = awwRFAll.predict_proba(utdAll)[:,1]

awwRFAcc = RandomForestClassifier(n_estimators=RFawwBestNumTrees_Acc)
awwRFAcc.fit(awwAcc,awwLabels)
RFutdPredAcc = awwRFAcc.predict_proba(utdAcc)[:,1]

awwRFEda = RandomForestClassifier(n_estimators=RFawwBestNumTrees_Eda)
awwRFEda.fit(awwEda,awwLabels)
RFutdPredEda = awwRFEda.predict_proba(utdEda)[:,1]

# Save the scores for further analysis
#np.save('utdPredAll_RF',RFutdPredAll)
#np.save('utdPredAcc_RF',RFutdPredAcc)
#np.save('utdPredEda_RF',RFutdPredEda)

print('Train on AWW, test on UTD')
print('RF AUC ALL: %f (%s)' % (roc_auc_score(utdLabels,RFutdPredAll),
                            RFawwGsAll.best_params_))
print('RF AUC ACC: %f (%s)' % (roc_auc_score(utdLabels,RFutdPredAcc),
                            RFawwGsAcc.best_params_))
print('RF AUC EDA: %f (%s)' % (roc_auc_score(utdLabels,RFutdPredEda),
                            RFawwGsEda.best_params_))


## kNN distance
# Perform grid search cross-validation on UTD data to identify best parameters
# and re-fit entire data set with best parameters. Test on AWW data.
maxNb = 30
utdCvAucAll = np.zeros((17,maxNb))
utdCvAucAcc = np.zeros((17,maxNb))
utdCvAucEda = np.zeros((17,maxNb))

fold = 0
for train,test in utdCv.split(utdAll[includeRowsTrain,:],
        utdLabels[includeRowsTrain],utdGroups[includeRowsTrain]):
    knnDAll = NearestNeighbors(n_neighbors=maxNb, algorithm='auto',
                               metric='euclidean')
    knnDAll.fit(utdAll[includeRowsTrain,:][train,:])
    distancesAll,indices = knnDAll.kneighbors(
        utdAll[includeRowsTrain,:][test,:])
    
    knnDAcc = NearestNeighbors(n_neighbors=maxNb, algorithm='auto', 
                               metric='euclidean')
    knnDAcc.fit(utdAcc[includeRowsTrain,:][train,:])
    distancesAcc,indices = knnDAcc.kneighbors(
        utdAcc[includeRowsTrain,:][test,:])

    knnDEda = NearestNeighbors(n_neighbors=maxNb, algorithm='auto',
                               metric='euclidean')
    knnDEda.fit(utdEda[includeRowsTrain,:][train,:])
    distancesEda,indices = knnDEda.kneighbors(
        utdEda[includeRowsTrain,:][test,:])
    
    for nNb in range(maxNb):
        utdCvAucAll[fold,nNb] = roc_auc_score(
            utdLabels[includeRowsTrain][test],distancesAll[:,nNb])
        utdCvAucAcc[fold,nNb] = roc_auc_score(
            utdLabels[includeRowsTrain][test],distancesAcc[:,nNb])
        utdCvAucEda[fold,nNb] = roc_auc_score(
            utdLabels[includeRowsTrain][test],distancesEda[:,nNb])
    fold += 1

utdBestNbAll = np.argmax(np.mean(utdCvAucAll,axis=0)) + 1
utdBestNbAcc = np.argmax(np.mean(utdCvAucAcc,axis=0)) + 1
utdBestNbEda = np.argmax(np.mean(utdCvAucEda,axis=0)) + 1

knnDAll = NearestNeighbors(n_neighbors=utdBestNbAll, algorithm='auto', 
                           metric='euclidean')
knnDAll.fit(utdAll)
awwdistancesAll,indices = knnDAll.kneighbors(awwAll)

knnDAcc = NearestNeighbors(n_neighbors=utdBestNbAcc, algorithm='auto', 
                           metric='euclidean')
knnDAcc.fit(utdAcc)
awwdistancesAcc,indices = knnDAcc.kneighbors(awwAcc)

knnDEda = NearestNeighbors(n_neighbors=utdBestNbEda, algorithm='auto', 
                           metric='euclidean')
knnDEda.fit(utdEda)
awwdistancesEda,indices = knnDEda.kneighbors(awwEda)

# Save the scores for further analysis
#np.save('awwPredAll_kNNdist',awwdistancesAll[:,utdBestNbAll-1])
#np.save('awwPredAcc_kNNdist',awwdistancesAcc[:,utdBestNbAcc-1])
#np.save('awwPredEda_kNNdist',awwdistancesEda[:,utdBestNbEda-1])

print('Train on UTD, test on AWW')       
print('kNNdist AUC ALL: %f (%d neighbors)' % (roc_auc_score(awwLabels,
      awwdistancesAll[:,utdBestNbAll-1]),utdBestNbAll))
print('kNNdist AUC ACC: %f (%d neighbors)' % (roc_auc_score(awwLabels,
      awwdistancesAcc[:,utdBestNbAcc-1]),utdBestNbAcc))
print('kNNdist AUC EDA: %f (%d neighbors)' % (roc_auc_score(awwLabels,
      awwdistancesEda[:,utdBestNbEda-1]),utdBestNbEda))

# Train on AWW, test on UTD
awwCvAucAll = np.zeros((17,maxNb))
awwCvAucAcc = np.zeros((17,maxNb))
awwCvAucEda = np.zeros((17,maxNb))

fold = 0
for train,test in awwCv.split(awwAll,awwLabels,awwGroups):
    knnDAll = NearestNeighbors(n_neighbors=maxNb, algorithm='auto',
                               metric='euclidean')
    knnDAll.fit(awwAll[train,:])
    distancesAll,indices = knnDAll.kneighbors(awwAll[test,:])
    
    knnDAcc = NearestNeighbors(n_neighbors=maxNb, algorithm='auto', 
                               metric='euclidean')
    knnDAcc.fit(awwAcc[train,:])
    distancesAcc,indices = knnDAcc.kneighbors(awwAcc[test,:])

    knnDEda = NearestNeighbors(n_neighbors=maxNb, algorithm='auto',
                               metric='euclidean')
    knnDEda.fit(awwEda[train,:])
    distancesEda,indices = knnDEda.kneighbors(awwEda[test,:])
    
    for nNb in range(maxNb):
        awwCvAucAll[fold,nNb] = roc_auc_score(
            awwLabels[test],distancesAll[:,nNb])
        awwCvAucAcc[fold,nNb] = roc_auc_score(
            awwLabels[test],distancesAcc[:,nNb])
        awwCvAucEda[fold,nNb] = roc_auc_score(
            awwLabels[test],distancesEda[:,nNb])
    fold += 1

awwBestNbAll = np.argmax(np.mean(awwCvAucAll,axis=0)) + 1
awwBestNbAcc = np.argmax(np.mean(awwCvAucAcc,axis=0)) + 1
awwBestNbEda = np.argmax(np.mean(awwCvAucEda,axis=0)) + 1

knnDAll = NearestNeighbors(n_neighbors=awwBestNbAll, algorithm='auto', 
                           metric='euclidean')
knnDAll.fit(awwAll)
utddistancesAll,indices = knnDAll.kneighbors(utdAll)

knnDAcc = NearestNeighbors(n_neighbors=awwBestNbAcc, algorithm='auto', 
                           metric='euclidean')
knnDAcc.fit(awwAcc)
utddistancesAcc,indices = knnDAcc.kneighbors(utdAcc)

knnDEda = NearestNeighbors(n_neighbors=awwBestNbEda, algorithm='auto', 
                           metric='euclidean')
knnDEda.fit(awwEda)
utddistancesEda,indices = knnDEda.kneighbors(utdEda)

# Save the scores for further analysis    
#np.save('utdPredAll_kNNdist',utddistancesAll[:,awwBestNbAll-1])
#np.save('utdPredAcc_kNNdist',utddistancesAcc[:,awwBestNbAcc-1])
#np.save('utdPredEda_kNNdist',utddistancesEda[:,awwBestNbEda-1])
#    
print('Train on AWW, test on UTD')       
print('kNNdist AUC ALL: %f (%d neighbors)' % (roc_auc_score(utdLabels,
      utddistancesAll[:,awwBestNbAll-1]),awwBestNbAll))
print('kNNdist AUC ACC: %f (%d neighbors)' % (roc_auc_score(utdLabels,
      utddistancesAcc[:,awwBestNbAcc-1]),awwBestNbAcc))
print('kNNdist AUC EDA: %f (%d neighbors)' % (roc_auc_score(utdLabels,
      utddistancesEda[:,awwBestNbEda-1]),awwBestNbEda))


## One Class SVM
# Perform grid search cross-validation on UTD data to identify best parameters
OneSVMparameters = {'gamma': 10. ** np.arange(-6,2), 'nu': 2. ** np.arange(-8,0)}

OneSVMutdGsAll = GridSearchCV(svm.OneClassSVM(),
                        OneSVMparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
OneSVMutdGsAll.fit(utdAll[includeRowsTrain,:],1-utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
OneSVMutdBestnu_All = OneSVMutdGsAll.best_params_['nu']
OneSVMutdbestgamma_All = OneSVMutdGsAll.best_params_['gamma']

OneSVMutdGsAcc = GridSearchCV(svm.OneClassSVM(),
                        OneSVMparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
OneSVMutdGsAcc.fit(utdAcc[includeRowsTrain,:],1-utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
OneSVMutdBestnu_Acc = OneSVMutdGsAcc.best_params_['nu']
OneSVMutdbestgamma_Acc = OneSVMutdGsAcc.best_params_['gamma']

OneSVMutdGsEda = GridSearchCV(svm.OneClassSVM(),
                        OneSVMparameters,'roc_auc',n_jobs=nJobs,cv=utdCv,refit=False,
                        verbose=1)
OneSVMutdGsEda.fit(utdEda[includeRowsTrain,:],1-utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
OneSVMutdBestnu_Eda = OneSVMutdGsEda.best_params_['nu']
OneSVMutdbestgamma_Eda = OneSVMutdGsEda.best_params_['gamma']

# Fit classifier with best parameters from grid search CV to entire UTD data
# (including excluded test subjects) and test on AWW data
utd1classSVMAll = svm.OneClassSVM(gamma=OneSVMutdbestgamma_All, nu=OneSVMutdBestnu_All, kernel='rbf')
utd1classSVMAll.fit(utdAll)
OneSVMawwPredAll = utd1classSVMAll.decision_function(awwAll)

utd1classSVMAcc = svm.OneClassSVM(gamma=OneSVMutdbestgamma_Acc, nu=OneSVMutdBestnu_Acc, kernel='rbf')
utd1classSVMAcc.fit(utdAcc)
OneSVMawwPredAcc = utd1classSVMAcc.decision_function(awwAcc)

utd1classSVMEda = svm.OneClassSVM(gamma=OneSVMutdbestgamma_Eda, nu=OneSVMutdBestnu_Eda, kernel='rbf')
utd1classSVMEda.fit(utdEda)
OneSVMawwPredEda = utd1classSVMEda.decision_function(awwEda)

# Save the scores for further analysis
#np.save('awwPredAll_1',OneSVMawwPredAll)
#np.save('awwPredAcc_1',OneSVMawwPredAcc)
#np.save('awwPredEda_1',OneSVMawwPredEda)

print('Train on UTD, test on AWW')
print('1classSVM AUC ALL: %f (%s)' % (roc_auc_score(1-awwLabels,OneSVMawwPredAll),
                            OneSVMutdGsAll.best_params_))
print('1classSVM AUC ACC: %f (%s)' % (roc_auc_score(1-awwLabels,OneSVMawwPredAcc),
                            OneSVMutdGsAcc.best_params_))
print('1classSVM AUC EDA: %f (%s)' % (roc_auc_score(1-awwLabels,OneSVMawwPredEda),
                            OneSVMutdGsEda.best_params_))

# Perform grid search cross-validation on AWW data to identify best parameters
# and re-fit entire data set with best parameters. Test on UTD data.
OneSVMawwGsAll = GridSearchCV(svm.OneClassSVM(),
                        OneSVMparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
OneSVMawwGsAll.fit(awwAll,1-awwLabels,awwGroups)
OneSVMawwBestnu_All = OneSVMawwGsAll.best_params_['nu']
OneSVMawwbestgamma_All = OneSVMawwGsAll.best_params_['gamma']

OneSVMawwGsAcc = GridSearchCV(svm.OneClassSVM(),
                        OneSVMparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
OneSVMawwGsAcc.fit(awwAcc,1-awwLabels,awwGroups)
OneSVMawwBestnu_Acc = OneSVMawwGsAcc.best_params_['nu']
OneSVMawwbestgamma_Acc = OneSVMawwGsAcc.best_params_['gamma']

OneSVMawwGsEda = GridSearchCV(svm.OneClassSVM(),
                        OneSVMparameters,'roc_auc',n_jobs=nJobs,cv=awwCv,refit=True,
                        verbose=1)
OneSVMawwGsEda.fit(awwEda,1-awwLabels,awwGroups)
OneSVMawwBestnu_Eda = OneSVMawwGsEda.best_params_['nu']
OneSVMawwbestgamma_Eda = OneSVMawwGsEda.best_params_['gamma']

awwOneSVMAll = svm.OneClassSVM(gamma=OneSVMawwbestgamma_All, nu=OneSVMawwBestnu_All, kernel='rbf')
awwOneSVMAll.fit(awwAll)
OneSVMutdPredAll = awwOneSVMAll.decision_function(utdAll)

awwOneSVMAcc = svm.OneClassSVM(gamma=OneSVMawwbestgamma_Acc, nu=OneSVMawwBestnu_Acc, kernel='rbf')
awwOneSVMAcc.fit(awwAcc)
OneSVMutdPredAcc = awwOneSVMAcc.decision_function(utdAcc)

awwOneSVMEda = svm.OneClassSVM(gamma=OneSVMawwbestgamma_Eda, nu=OneSVMawwBestnu_Eda, kernel='rbf')
awwOneSVMEda.fit(awwEda)
OneSVMutdPredEda = awwOneSVMEda.decision_function(utdEda)

# Save the scores for further analysis
#np.save('utdPredAll_1',OneSVMutdPredAll)
#np.save('utdPredAcc_1',OneSVMutdPredAcc)
#np.save('utdPredEda_1',OneSVMutdPredEda)

print('Train on AWW, test on UTD')
print('1classSVM AUC ALL: %f (%s)' % (roc_auc_score(1-utdLabels,OneSVMutdPredAll),
                            OneSVMawwGsAll.best_params_))
print('1classSVM AUC ACC: %f (%s)' % (roc_auc_score(1-utdLabels,OneSVMutdPredAcc),
                            OneSVMawwGsAcc.best_params_))
print('1classSVM AUC EDA: %f (%s)' % (roc_auc_score(1-utdLabels,OneSVMutdPredEda),
                            OneSVMawwGsEda.best_params_))


## Isolation Forest
# Perform grid search cross-validation on UTD data to identify best parameters
IFparameters = {'n_estimators': 10*np.arange(1,21)}

IFutdGsAll = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=24,cv=utdCv,refit=False,
                        verbose=1)
IFutdGsAll.fit(utdAll[includeRowsTrain,:],1-utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
IFutdBestNumTrees_All = IFutdGsAll.best_params_['n_estimators']

IFutdGsAcc = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=24,cv=utdCv,refit=False,
                        verbose=1)
IFutdGsAcc.fit(utdAcc[includeRowsTrain,:],1-utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
IFutdBestNumTrees_Acc = IFutdGsAcc.best_params_['n_estimators']

IFutdGsEda = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=24,cv=utdCv,refit=False,
                        verbose=1)
IFutdGsEda.fit(utdEda[includeRowsTrain,:],1-utdLabels[includeRowsTrain],
             utdGroups[includeRowsTrain])
IFutdBestNumTrees_Eda = IFutdGsEda.best_params_['n_estimators']

# Fit classifier with best parameters from grid search CV to entire UTD data
# (including excluded test subjects) and test on AWW data
utdIFAll = IsolationForest(n_estimators=IFutdBestNumTrees_All)
utdIFAll.fit(utdAll)
IFawwPredAll = utdIFAll.decision_function(awwAll)

utdIFAcc = IsolationForest(n_estimators=IFutdBestNumTrees_Acc)
utdIFAcc.fit(utdAcc)
IFawwPredAcc = utdIFAcc.decision_function(awwAcc)

utdIFEda = IsolationForest(n_estimators=IFutdBestNumTrees_Eda)
utdIFEda.fit(utdEda)
IFawwPredEda = utdIFEda.decision_function(awwEda)

# Save the scores for further analysis
#np.save('awwPredAll_IF',IFawwPredAll)
#np.save('awwPredAcc_IF',IFawwPredAcc)
#np.save('awwPredEda_IF',IFawwPredEda)

print('Train on UTD, test on AWW')
print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-awwLabels,IFawwPredAll),
                            IFutdGsAll.best_params_))
print('IF AUC ACC: %f (%s)' % (roc_auc_score(1-awwLabels,IFawwPredAcc),
                            IFutdGsAcc.best_params_))
print('IF AUC EDA: %f (%s)' % (roc_auc_score(1-awwLabels,IFawwPredEda),
                            IFutdGsEda.best_params_))

# Perform grid search cross-validation on AWW data to identify best parameters
# and re-fit entire data set with best parameters. Test on UTD data.
IFawwGsAll = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=24,cv=awwCv,refit=True,
                        verbose=1)
IFawwGsAll.fit(awwAll,1-awwLabels,awwGroups)
IFawwBestNumTrees_All = IFawwGsAll.best_params_['n_estimators']

IFawwGsAcc = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=24,cv=awwCv,refit=True,
                        verbose=1)
IFawwGsAcc.fit(awwAcc,1-awwLabels,awwGroups)
IFawwBestNumTrees_Acc = IFawwGsAcc.best_params_['n_estimators']

IFawwGsEda = GridSearchCV(IsolationForest(),
                        IFparameters,'roc_auc',n_jobs=24,cv=awwCv,refit=True,
                        verbose=1)
IFawwGsEda.fit(awwEda,1-awwLabels,awwGroups)
IFawwBestNumTrees_Eda = IFawwGsEda.best_params_['n_estimators']

awwIFAll = IsolationForest(n_estimators=IFawwBestNumTrees_All)
awwIFAll.fit(awwAll)
IFutdPredAll = awwIFAll.decision_function(utdAll)

awwIFAcc = IsolationForest(n_estimators=IFawwBestNumTrees_Acc)
awwIFAcc.fit(awwAcc)
IFutdPredAcc = awwIFAcc.decision_function(utdAcc)

awwIFEda = IsolationForest(n_estimators=IFawwBestNumTrees_Eda)
awwIFEda.fit(awwEda)
IFutdPredEda = awwIFEda.decision_function(utdEda)

# Save the scores for further analysis
#np.save('utdPredAll_IF',IFutdPredAll)
#np.save('utdPredAcc_IF',IFutdPredAcc)
#np.save('utdPredEda_IF',IFutdPredEda)

print('Train on AWW, test on UTD')
print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-utdLabels,IFutdPredAll),
                            IFawwGsAll.best_params_))
print('IF AUC ACC: %f (%s)' % (roc_auc_score(1-utdLabels,IFutdPredAcc),
                            IFawwGsAcc.best_params_))
print('IF AUC EDA: %f (%s)' % (roc_auc_score(1-utdLabels,IFutdPredEda),
                            IFawwGsEda.best_params_))





