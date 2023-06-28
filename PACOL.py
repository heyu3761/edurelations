# -*- coding: utf-8 -*-
from alipy import ToolBox
import copy
import numpy as np
import  pandas  as pd
from sklearn.ensemble import GradientBoostingClassifier
import alipy
import random
from sklearn import metrics
from pandas import DataFrame
from  alipy.query_strategy import QueryInstanceUncertainty
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import  train_test_split




def cos(array1, array2):
    norm1 = np.sqrt(sum(list(map(lambda x: np.power(x, 2), array1))))
    norm2 = np.sqrt(sum(list(map(lambda x: np.power(x, 2), array2))))
    return sum([array1[i]*array2[i] for i in range(0, len(array1))]) / (norm1 * norm2)



def selectInstance(label_ind, unlabel_ind, model):

    ret1 = unlabel_ind.index[0]
    ret2 = int(dic[ret1])
    maxG = 0


    y_p = model.predict(X)
    pro = model.predict_proba(X)
    ui = unlabel_ind.index

    for u in unlabel_ind.index:
        p = int(dic[u])
        y1 = y_p[u]
        y2 = y_p[p]
        p1 = pro[u]
        p2 = pro[p]
        p1 = max(p1)
        p2 = max(p2)

        if y1*p1+y2*p2!=0:
            sim = 0
            for sinu in ui:
                sim1 = cos(X[u],X[sinu])
                sim2 = cos(X[p],X[sinu])
                sim = sim + sim1 + sim2
            ans = abs(y1*p1+y2*p2)*sim
            if ans > maxG:
                maxG = ans
                ret1 = u
                ret2 = p
    return ret1,ret2


excelFile = os.path.join('./','math.xlsx')
df=pd.read_excel(excelFile)
dataset = df.values

dataset[np.isnan(dataset)] = 0
dic = dataset[:,22]

pair_index = dataset[:,0:2]
feature = dataset[:,2:20]
label = dataset[:,20]
X, y = feature, label
index = len(y)
print(index)

x1 = dataset[0]
x2 = dataset[int(dic[0])]

dic_sim ={}

alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='./my')

# Split data
alibox.split_AL(test_ratio=0.2, initial_label_rate=0.1, split_count=1)

print("init model ")
model = GradientBoostingClassifier(n_estimators=20,learning_rate=0.1,min_samples_split=300,min_samples_leaf=40,max_depth=13,max_features='sqrt')


unc_result = []
auc_ret = []
print("start active learning")
for round in range(1):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)

    # Set initial performance point
    model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
    pred = model.predict(X[test_idx, :])
    y_predict_prob = model.predict_proba(X[test_idx, :])
    test_auc = metrics.roc_auc_score(y[test_idx],y_predict_prob,multi_class='ovr')
    saver.set_initial_point(test_auc)

    while len(unlab_ind.index)>0:
        select_ind1, select_ind2 = selectInstance(label_ind, unlab_ind, model=model)

        if select_ind1 in unlab_ind.index:
            label_ind.update(select_ind1)
            unlab_ind.difference_update(select_ind1)
        else:
            select_ind =  selectInstance(label_ind, unlab_ind, model=model, batch_size=1)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
        if select_ind2 in unlab_ind.index:
            #print(select_ind2 in unlab_ind.index)
            label_ind.update(select_ind2)
            unlab_ind.difference_update(select_ind2)
            #print(select_ind2 in unlab_ind.index)
        else:
            select_ind =  selectInstance.select(label_ind, unlab_ind, model=model, batch_size=1)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        y_predict_prob = model.predict_proba(X[test_idx, :])
        test_auc = metrics.roc_auc_score(y[test_idx],y_predict_prob,multi_class='ovr')
        auc_ret.append(copy.deepcopy(test_auc))
        # Save intermediate results to file
        st = alibox.State(select_index=select_ind1, performance=test_auc)
        saver.add_state(st)
        st = alibox.State(select_index=select_ind2, performance=test_auc)
        saver.add_state(st)
       # saver.save()

        # Passing the current progress to stopping criterion object
     #   stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
   # stopping_criterion.reset()
    unc_result.append(copy.deepcopy(saver))
df=DataFrame(auc_ret)
df.to_excel('./my/math_all_simall_ans.xlsx')
print('./my/math_all_simall_ans.xlsx')
analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
print(analyser)
analyser.plot_learning_curves(title='Example of AL', std_area=True)