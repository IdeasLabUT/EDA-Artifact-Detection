# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:56:29 2017

@author: yzhang17
"""

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier,IsolationForest

dataPath = 'UTDallas/'
dataName = 'UTD'
nJobs = 12  # Number of cores to use

# Load feature matrices, labels, and groups (denoting which labeled time
# segment each row of the feature matrix comes from)
featuresAll = np.loadtxt(dataPath+dataName+'_all.csv',delimiter=',')
featuresAcc = np.loadtxt(dataPath+dataName+'_acc.csv',delimiter=',')
featuresEda = np.loadtxt(dataPath+dataName+'_eda.csv',delimiter=',')
labels = np.loadtxt(dataPath+dataName+'_label.csv')
groups = np.loadtxt(dataPath+dataName+'_groups.csv')
# Indicates the subjects that have no MAs, in order to exclude them during grid search
includeRowsTrain = np.logical_and(
    np.logical_and(np.where(groups!=5,True,False),
    np.where(groups!=17,True,False)),np.where(groups!=18,True,False))

# Leave-one-group-out cross-validation
cv = LeaveOneGroupOut() 

## Svm
# Parameter tuning by grid search
regParamC = 10. ** np.arange(-3, 5)
regParamG = 10. ** np.arange(-10, 1)
parameters = {'gamma': regParamG, 'C': regParamC}
              
svmgsAll = GridSearchCV(svm.SVC(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
svmgsAll.fit(featuresAll[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
svmbestgamma_All = svmgsAll.best_params_['gamma']
svmbestC_All = svmgsAll.best_params_['C']

svmgsAcc = GridSearchCV(svm.SVC(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
svmgsAcc.fit(featuresAcc[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
svmbestgamma_Acc = svmgsAcc.best_params_['gamma']
svmbestC_Acc = svmgsAcc.best_params_['C']

svmgsEda = GridSearchCV(svm.SVC(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
svmgsEda.fit(featuresEda[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
svmbestgamma_Eda = svmgsEda.best_params_['gamma']
svmbestC_Eda = svmgsEda.best_params_['C']

svmpredAll = np.zeros(np.shape(labels))
svmpredAcc = np.zeros(np.shape(labels))
svmpredEda = np.zeros(np.shape(labels))
for train, test in cv.split(featuresAll,labels,groups):
    svmAll = svm.SVC(gamma=svmbestgamma_All, C=svmbestC_All, kernel='rbf',probability=True)
    svmAll.fit(featuresAll[train,:],labels[train])
    svmpredAll[test] = svmAll.predict_proba(featuresAll[test,:])[:,1]

    svmAcc = svm.SVC(gamma=svmbestgamma_Acc, C=svmbestC_Acc, kernel='rbf',probability=True)
    svmAcc.fit(featuresAcc[train,:],labels[train])
    svmpredAcc[test] = svmAcc.predict_proba(featuresAcc[test,:])[:,1]

    svmEda = svm.SVC(gamma=svmbestgamma_Eda, C=svmbestC_Eda, kernel='rbf',probability=True)
    svmEda.fit(featuresEda[train,:],labels[train])
    svmpredEda[test] = svmEda.predict_proba(featuresEda[test,:])[:,1]

# Save the scores for further analysis
#np.save('svmpredAllScores_UTD',svmpredAll)
#np.save('svmpredAccScores_UTD',svmpredAcc)
#np.save('svmpredEdaScores_UTD',svmpredEda)

print('SVM AUC ALL: %f (%s)' % (roc_auc_score(labels,svmpredAll),svmgsAll.best_params_))
print('SVM AUC ACC: %f (%s)' % (roc_auc_score(labels,svmpredAcc),svmgsAcc.best_params_))
print('SVM AUC EDA: %f (%s)' % (roc_auc_score(labels,svmpredEda),svmgsEda.best_params_))

## Logistic
# Parameter tuning by grid search
regParam = 10.0**-np.arange(-3,7)
parameters = {'C': regParam}

gsAll = GridSearchCV(LogisticRegression(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
gsAll.fit(featuresAll[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
bestC_All = gsAll.best_params_['C']

gsAcc = GridSearchCV(LogisticRegression(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
gsAcc.fit(featuresAcc[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
bestC_Acc = gsAcc.best_params_['C']

gsEda = GridSearchCV(LogisticRegression(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
gsEda.fit(featuresEda[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
bestC_Eda = gsEda.best_params_['C']

LogisticpredAll = np.zeros(np.shape(labels))
LogisticpredAcc = np.zeros(np.shape(labels))
LogisticpredEda = np.zeros(np.shape(labels))
for train, test in cv.split(featuresAll,labels,groups):
    lrAll = LogisticRegression(C=bestC_All)
    lrAll.fit(featuresAll[train,:],labels[train])
    LogisticpredAll[test] = lrAll.predict_proba(featuresAll[test,:])[:,1]

    lrAcc = LogisticRegression(C=bestC_Acc)
    lrAcc.fit(featuresAcc[train,:],labels[train])
    LogisticpredAcc[test] = lrAcc.predict_proba(featuresAcc[test,:])[:,1]

    lrEda = LogisticRegression(C=bestC_Eda)
    lrEda.fit(featuresEda[train,:],labels[train])
    LogisticpredEda[test] = lrEda.predict_proba(featuresEda[test,:])[:,1]

# Save the scores for further analysis
#np.save('LogisticpredAllScores_UTD',LogisticpredAll)
#np.save('LogisticpredAccScores_UTD',LogisticpredAcc)
#np.save('LogisticpredEdaScores_UTD',LogisticpredEda)

print('LR AUC ALL: %f (%s)' % (roc_auc_score(labels,LogisticpredAll),gsAll.best_params_))
print('LR AUC ACC: %f (%s)' % (roc_auc_score(labels,LogisticpredAcc),gsAcc.best_params_))
print('LR AUC EDA: %f (%s)' % (roc_auc_score(labels,LogisticpredEda),gsEda.best_params_))

## kNN classify
# Parameter tuning by grid search
parameters = {'n_neighbors': 2 ** np.arange(0, 9)}

kNNgsAll = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
kNNgsAll.fit(featuresAll[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
bestneighbors_All = kNNgsAll.best_params_['n_neighbors']

kNNgsAcc = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
kNNgsAcc.fit(featuresAcc[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
bestneighbors_Acc = kNNgsAcc.best_params_['n_neighbors']

kNNgsEda = GridSearchCV(KNeighborsClassifier(algorithm='auto', metric='euclidean'),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
kNNgsEda.fit(featuresEda[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
bestneighbors_Eda = kNNgsEda.best_params_['n_neighbors']              

knnCpredAll = np.zeros(np.shape(labels))
knnCpredAcc = np.zeros(np.shape(labels))
knnCpredEda = np.zeros(np.shape(labels))              
for train, test in cv.split(featuresAll,labels,groups):
    knnCAll = KNeighborsClassifier(n_neighbors=bestneighbors_All, algorithm='auto', metric='euclidean')
    knnCAll.fit(featuresAll[train,:],labels[train])
    knnCpredAll[test] = knnCAll.predict_proba(featuresAll[test,:])[:,1]

    knnCAcc = KNeighborsClassifier(n_neighbors=bestneighbors_Acc, algorithm='auto', metric='euclidean')
    knnCAcc.fit(featuresAcc[train,:],labels[train])
    knnCpredAcc[test] = knnCAcc.predict_proba(featuresAcc[test,:])[:,1]

    knnCEda = KNeighborsClassifier(n_neighbors=bestneighbors_Eda, algorithm='auto', metric='euclidean')
    knnCEda.fit(featuresEda[train,:],labels[train])
    knnCpredEda[test] = knnCEda.predict_proba(featuresEda[test,:])[:,1]

# Save the scores for further analysis
#np.save('knnCpredAllScores_UTD',knnCpredAll)
#np.save('knnCpredAccScores_UTD',knnCpredAcc)
#np.save('knnCpredEdaScores_UTD',knnCpredEda)

print('kNNclass AUC ALL: %f (%s)' % (roc_auc_score(labels,knnCpredAll),kNNgsAll.best_params_))
print('kNNclass AUC ACC: %f (%s)' % (roc_auc_score(labels,knnCpredAcc),kNNgsAcc.best_params_))
print('kNNclass AUC EDA: %f (%s)' % (roc_auc_score(labels,knnCpredEda),kNNgsEda.best_params_))

## Random Forest
# Parameter tuning by grid search
RFparameters = {'n_estimators': 10*np.arange(1,21)}

RFgsAll = GridSearchCV(RandomForestClassifier(),
                     RFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
RFgsAll.fit(featuresAll[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
RFbestNumTreesAll = RFgsAll.best_params_['n_estimators']

RFgsAcc = GridSearchCV(RandomForestClassifier(),
                     RFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
RFgsAcc.fit(featuresAcc[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
RFbestNumTreesAcc = RFgsAcc.best_params_['n_estimators']

RFgsEda = GridSearchCV(RandomForestClassifier(),
                     RFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
RFgsEda.fit(featuresEda[includeRowsTrain,:],labels[includeRowsTrain],groups[includeRowsTrain])
RFbestNumTreesEda = RFgsEda.best_params_['n_estimators']

RFpredAll = np.zeros(np.shape(labels))
RFpredAcc = np.zeros(np.shape(labels))
RFpredEda = np.zeros(np.shape(labels))
for train, test in cv.split(featuresAll,labels,groups):
    RFAll = RandomForestClassifier(n_estimators=RFbestNumTreesAll)
    RFAll.fit(featuresAll[train,:],labels[train])
    RFpredAll[test] = RFAll.predict_proba(featuresAll[test,:])[:,1]

    RFAcc = RandomForestClassifier(n_estimators=RFbestNumTreesAcc)
    RFAcc.fit(featuresAcc[train,:],labels[train])
    RFpredAcc[test] = RFAcc.predict_proba(featuresAcc[test,:])[:,1]

    RFEda = RandomForestClassifier(n_estimators=RFbestNumTreesEda)
    RFEda.fit(featuresEda[train,:],labels[train])
    RFpredEda[test] = RFEda.predict_proba(featuresEda[test,:])[:,1]

# Save the scores for further analysis
#np.save('RandomForestpredAllScores_utd',RFpredAll)
#np.save('RandomForestpredAccScores_utd',RFpredAcc)
#np.save('RandomForestpredEdaScores_utd',RFpredEda)

print('RF AUC ALL: %f (%s)' % (roc_auc_score(labels,RFpredAll),RFgsAll.best_params_))
print('RF AUC ACC: %f (%s)' % (roc_auc_score(labels,RFpredAcc),RFgsAcc.best_params_))
print('RF AUC EDA: %f (%s)' % (roc_auc_score(labels,RFpredEda),RFgsEda.best_params_))

## kNN distance
# Parameter tuning by grid search
maxNb = 30
awwCv = LeaveOneGroupOut()
awwCvAucAll = np.zeros((17,maxNb))
awwCvAucAcc = np.zeros((17,maxNb))
awwCvAucEda = np.zeros((17,maxNb))

fold = 0
for train,test in awwCv.split(featuresAll[includeRowsTrain,:],
        labels[includeRowsTrain],groups[includeRowsTrain]):
    knnDAll = NearestNeighbors(n_neighbors=maxNb, algorithm='auto',
                               metric='euclidean')
    knnDAll.fit(featuresAll[includeRowsTrain,:][train,:])
    distancesAll,indices = knnDAll.kneighbors(featuresAll[test,:])
    
    knnDAcc = NearestNeighbors(n_neighbors=maxNb, algorithm='auto', 
                               metric='euclidean')
    knnDAcc.fit(featuresAcc[includeRowsTrain,:][train,:])
    distancesAcc,indices = knnDAcc.kneighbors(featuresAcc[test,:])

    knnDEda = NearestNeighbors(n_neighbors=maxNb, algorithm='auto',
                               metric='euclidean')
    knnDEda.fit(featuresEda[includeRowsTrain,:][train,:])
    distancesEda,indices = knnDEda.kneighbors(featuresEda[test,:])
    
    for nNb in range(maxNb):
        awwCvAucAll[fold,nNb] = roc_auc_score(
            labels[includeRowsTrain][test],distancesAll[:,nNb])
        awwCvAucAcc[fold,nNb] = roc_auc_score(
            labels[includeRowsTrain][test],distancesAcc[:,nNb])
        awwCvAucEda[fold,nNb] = roc_auc_score(
            labels[includeRowsTrain][test],distancesEda[:,nNb])
    fold += 1

awwBestNbAll = np.argmax(np.mean(awwCvAucAll,axis=0)) + 1
awwBestNbAcc = np.argmax(np.mean(awwCvAucAcc,axis=0)) + 1
awwBestNbEda = np.argmax(np.mean(awwCvAucEda,axis=0)) + 1

distancesAll = np.zeros((np.size(labels),awwBestNbAll))
distancesAcc = np.zeros((np.size(labels),awwBestNbAcc))
distancesEda = np.zeros((np.size(labels),awwBestNbEda))
for train, test in cv.split(featuresAll,labels,groups):
    knnDAll = NearestNeighbors(n_neighbors=awwBestNbAll, algorithm='auto', metric='euclidean')
    knnDAll.fit(featuresAll[train,:])
    distancesAll[test],indices = knnDAll.kneighbors(featuresAll[test,:])

    knnDAcc = NearestNeighbors(n_neighbors=awwBestNbAcc, algorithm='auto', metric='euclidean')
    knnDAcc.fit(featuresAcc[train,:])
    distancesAcc[test],indices = knnDAcc.kneighbors(featuresAcc[test,:])

    knnDEda = NearestNeighbors(n_neighbors=awwBestNbEda, algorithm='auto', metric='euclidean')
    knnDEda.fit(featuresEda[train,:])
    distancesEda[test],indices = knnDEda.kneighbors(featuresEda[test,:])

# Save the scores for further analysis
#np.save('distancespredAllScores_UTD',distancesAll[:,awwBestNbAll-1])
#np.save('distancespredAccScores_UTD',distancesAcc[:,awwBestNbAcc-1])
#np.save('distancespredEdaScores_UTD',distancesEda[:,awwBestNbEda-1])

print('kNNdist AUC ALL: %f (%d neighbors)' % (roc_auc_score(labels,
      distancesAll[:,awwBestNbAll-1]),awwBestNbAll))
print('kNNdist AUC ACC: %f (%d neighbors)' % (roc_auc_score(labels,
      distancesAcc[:,awwBestNbAcc-1]),awwBestNbAcc))
print('kNNdist AUC EDA: %f (%d neighbors)' % (roc_auc_score(labels,
      distancesEda[:,awwBestNbEda-1]),awwBestNbEda))

## 1 class Svm
# Parameter tuning by grid search
regParamG = 10. ** np.arange(-6, 2)
regParamNu = 2. ** np.arange(-8,0)
parameters = {'gamma': regParamG, 'nu': regParamNu}

OnesvmgsAll = GridSearchCV(svm.OneClassSVM(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
OnesvmgsAll.fit(featuresAll[includeRowsTrain,:],1-labels[includeRowsTrain],groups[includeRowsTrain])
Onesvmbestgamma_All = OnesvmgsAll.best_params_['gamma']
Onesvmbestnu_All = OnesvmgsAll.best_params_['nu']

OnesvmgsAcc = GridSearchCV(svm.OneClassSVM(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
OnesvmgsAcc.fit(featuresAcc[includeRowsTrain,:],1-labels[includeRowsTrain],groups[includeRowsTrain])
Onesvmbestgamma_Acc = OnesvmgsAcc.best_params_['gamma']
Onesvmbestnu_Acc = OnesvmgsAcc.best_params_['nu']

OnesvmgsEda = GridSearchCV(svm.OneClassSVM(),
                     parameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
OnesvmgsEda.fit(featuresEda[includeRowsTrain,:],1-labels[includeRowsTrain],groups[includeRowsTrain])
Onesvmbestgamma_Eda = OnesvmgsEda.best_params_['gamma']
Onesvmbestnu_Eda = OnesvmgsEda.best_params_['nu']

OnesvmpredAll = np.zeros(np.shape(labels))
OnesvmpredAcc = np.zeros(np.shape(labels))
OnesvmpredEda = np.zeros(np.shape(labels))
for train, test in cv.split(featuresAll,labels,groups):
    OnesvmAll = svm.OneClassSVM(gamma=Onesvmbestgamma_All, nu=Onesvmbestnu_All, kernel='rbf')
    OnesvmAll.fit(featuresAll[train,:])
    OnesvmpredAll[test] = OnesvmAll.decision_function(featuresAll[test,:]).ravel()

    OnesvmAcc = svm.OneClassSVM(gamma=Onesvmbestgamma_Acc, nu=Onesvmbestnu_Acc, kernel='rbf')
    OnesvmAcc.fit(featuresAcc[train,:])
    OnesvmpredAcc[test] = OnesvmAcc.decision_function(featuresAcc[test,:]).ravel()

    OnesvmEda = svm.OneClassSVM(gamma=Onesvmbestgamma_Eda, nu=Onesvmbestnu_Eda, kernel='rbf')
    OnesvmEda.fit(featuresEda[train,:])
    OnesvmpredEda[test] = OnesvmEda.decision_function(featuresEda[test,:]).ravel()

# Save the scores for further analysis
#np.save('OnesvmpredAllScores_UTD',OnesvmpredAll)
#np.save('OnesvmpredAccScores_UTD',OnesvmpredAcc)
#np.save('OnesvmpredEdaScores_UTD',OnesvmpredEda)

print('OneSVM AUC ALL: %f (%s)' % (roc_auc_score(1-labels,OnesvmpredAll),OnesvmgsAll.best_params_))
print('OneSVM AUC ACC: %f (%s)' % (roc_auc_score(1-labels,OnesvmpredAcc),OnesvmgsAcc.best_params_))
print('OneSVM AUC EDA: %f (%s)' % (roc_auc_score(1-labels,OnesvmpredEda),OnesvmgsEda.best_params_))

## Isolation Forest
# Parameter tuning by grid search
IFparameters = {'n_estimators': 10*np.arange(1,21)}

IFgsAll = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
IFgsAll.fit(featuresAll[includeRowsTrain,:],1-labels[includeRowsTrain],groups[includeRowsTrain])
bestNumTreesAll = IFgsAll.best_params_['n_estimators']

IFgsAcc = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
IFgsAcc.fit(featuresAcc[includeRowsTrain,:],1-labels[includeRowsTrain],groups[includeRowsTrain])
bestNumTreesAcc = IFgsAcc.best_params_['n_estimators']

IFgsEda = GridSearchCV(IsolationForest(),
                     IFparameters,'roc_auc',n_jobs=nJobs,cv=cv,refit=False,
                     verbose=1)
IFgsEda.fit(featuresEda[includeRowsTrain,:],1-labels[includeRowsTrain],groups[includeRowsTrain])
bestNumTreesEda = IFgsEda.best_params_['n_estimators']

IFpredAll = np.zeros(np.shape(labels))
IFpredAcc = np.zeros(np.shape(labels))
IFpredEda = np.zeros(np.shape(labels))
for train, test in cv.split(featuresAll,labels,groups):
    IFAll = IsolationForest(n_estimators=bestNumTreesAll)
    IFAll.fit(featuresAll[train,:])
    IFpredAll[test] = IFAll.decision_function(featuresAll[test,:])

    IFAcc = IsolationForest(n_estimators=bestNumTreesAcc)
    IFAcc.fit(featuresAcc[train,:])
    IFpredAcc[test] = IFAcc.decision_function(featuresAcc[test,:])

    IFEda = IsolationForest(n_estimators=bestNumTreesEda)
    IFEda.fit(featuresEda[train,:])
    IFpredEda[test] = IFEda.decision_function(featuresEda[test,:])

# Save the scores for further analysis
#np.save('IsolationForestpredAllScores_UTD',IFpredAll)
#np.save('IsolationForestpredAccScores_UTD',IFpredAcc)
#np.save('IsolationForestpredEdaScores_UTD',IFpredEda)

print('IF AUC ALL: %f (%s)' % (roc_auc_score(1-labels,IFpredAll),IFgsAll.best_params_))
print('IF AUC ACC: %f (%s)' % (roc_auc_score(1-labels,IFpredAcc),IFgsAcc.best_params_))
print('IF AUC EDA: %f (%s)' % (roc_auc_score(1-labels,IFpredEda),IFgsEda.best_params_))
