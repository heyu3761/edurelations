# -*- coding: utf-8 -*-
import numpy as np
import  pandas  as pd
from sklearn.naive_bayes import GaussianNB
#from sklearn.preprocessing import label_binarize

from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
import sklearn.ensemble as se
from sklearn import metrics

#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def score( y_true, y_predict_prob):
    return metrics.roc_auc_score(y_true,y_predict_prob,multi_class='ovr')
    

df=pd.read_excel('math1.xlsx')
dataset = df.values

# 补充0
dataset[np.isnan(dataset)] = 0
print(dataset.shape)

dataset = dataset


feature = dataset[:,0:18]
label = dataset[:,18]
print(feature.shape,label.shape)

X_train,X_test,y_train,y_test = train_test_split(feature,label,test_size=0.2,random_state=33)


# NB
mnb = GaussianNB()   # 使用默认配置初始化朴素贝叶斯
mnb.fit(X_train,y_train)    # 利用训练数据对模型参数进行估计
y_predict = mnb.predict(X_test)     # 对参数进行预测
print('The Accuracy of Naive Bayes Classifier is:', mnb.score(X_test,y_test))
y_predict_prob = mnb.predict_proba(X_test)

test_auc = metrics.roc_auc_score(y_test,y_predict_prob,multi_class='ovr')
print("auc = ",test_auc)
print(classification_report(y_test, y_predict))


# RF 
model = se.RandomForestClassifier(max_depth=13, n_estimators=60, criterion ="entropy",min_samples_split=300,min_samples_leaf=30)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print('The Accuracy of RF Classifier is:', model.score(X_test,y_test))
y_predict_prob = model.predict_proba(X_test)


test_auc = metrics.roc_auc_score(y_test,y_predict_prob,multi_class='ovr')
print("auc = ",test_auc)
print(classification_report(y_test, y_predict))


# SVM 
clf = svm.SVC( probability=True)
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
print('The Accuracy of SVM Classifier is:', clf.score(X_test,y_test))
y_predict_prob = clf.predict_proba(X_test)

test_auc = metrics.roc_auc_score(y_test,y_predict_prob,multi_class='ovr')
print("auc = ",test_auc)
print(classification_report(y_test, y_predict))

# LR
lr = LogisticRegression(C=1000.0, random_state=0,class_weight={-1:0.45,1:0.45,0:0.1})
lr.fit(X_train,y_train)
y_predict = lr.predict(X_test)
print('The Accuracy of LogisticRegression Classifier is:', lr.score(X_test,y_test))
y_predict_prob = lr.predict_proba(X_test)

test_auc = metrics.roc_auc_score(y_test,y_predict_prob,multi_class='ovr')
print("auc = ",test_auc)
print(classification_report(y_test, y_predict))


gbm0= GradientBoostingClassifier(n_estimators=20,learning_rate=0.1,min_samples_split=300,min_samples_leaf=40,max_depth=13,max_features='sqrt')

gbm0.fit(X_train,y_train)
y_predict = gbm0.predict(X_test)
print('The Accuracy of GBDT Classifier is:', gbm0.score(X_test,y_test))
y_predict_prob = gbm0.predict_proba(X_test)


test_auc = metrics.roc_auc_score(y_test,y_predict_prob,multi_class='ovr')
print("auc = ",test_auc)
print(classification_report(y_test, y_predict))


gbm0= GradientBoostingClassifier(n_estimators=30)
gbm0.fit(X_train,y_train)
y_predict = gbm0.predict(X_test)
print('The Accuracy of GBDT Classifier is:', gbm0.score(X_test,y_test))
y_predict_prob = gbm0.predict_proba(X_test)


test_auc = metrics.roc_auc_score(y_test,y_predict_prob,multi_class='ovr')
print("auc = ",test_auc)
print(classification_report(y_test, y_predict))

'''
scoring = {'roc': make_scorer(score,greater_is_better=True,needs_proba = True)}
param_grid = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200), 'n_estimators':range(20,80,20), 'min_samples_leaf':range(10,50,10)}
forest_reg = se.GradientBoostingClassifier(learning_rate=0.1)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring=scoring,refit='roc')
grid_search.fit(feature,list(label.astype('int')))
print(grid_search.best_params_, grid_search.best_score_)

'''