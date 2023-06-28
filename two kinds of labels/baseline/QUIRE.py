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
from  alipy.query_strategy import QueryInstanceQUIRE
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import  train_test_split

def cos(array1, array2):
    norm1 = np.sqrt(sum(list(map(lambda x: np.power(x, 2), array1))))
    norm2 = np.sqrt(sum(list(map(lambda x: np.power(x, 2), array2))))
    return sum([array1[i]*array2[i] for i in range(0, len(array1))]) / (norm1 * norm2)
df=pd.read_excel('../math.xlsx')
dataset = df.values

# 补充0
dataset[np.isnan(dataset)] = 0
dic = dataset[:,22]

pair_index = dataset[:,0:2]
feature = dataset[:,2:20]
label = dataset[:,20]
X, y = feature, label
index = len(y)
print(index)



def selectInstance(label_ind, unlabel_ind, model):
    # 预测未标记实例的标签
  #  y_pred = model.predict(X[unlabel_ind.index,:])
    
        
    ret1 = unlabel_ind.index[0]
    ret2 = int(dic[ret1])
    maxG = 0
    
    # 遍历所有实例，计算知识点对之间的差距，记录下最大的那个
    
    pro = model.predict_proba(X)
    
    for u in unlabel_ind.index:
        
      
        p1 = pro[u]
       
        p1 = max(p1)

        ans = p1
        if ans >= maxG:
            maxG = ans
            ret2 = ret1
            ret1 = u
    return ret1,ret2     
    



alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='./4')

# Split data
alibox.split_AL(test_ratio=0.2, initial_label_rate=0.1, split_count=1)

# Use the default Logistic Regression classifier
#model = alibox.get_default_model()
print("init model ")
model = GradientBoostingClassifier(n_estimators=20,learning_rate=0.1,min_samples_split=300,min_samples_leaf=40,max_depth=13,max_features='sqrt')

print("finish init model ")
# The cost budget is 50 times querying
#stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 20000)
uncertainStrategy = QueryInstanceQUIRE(X=X, y=y, measure='least_confident')
# Use pre-defined strategy
#uncertainStrategy = alibox.get_query_strategy(strategy_name='QueryInstanceQUIRE')
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
 #   accuracy = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=pred, performance_metric='f1_score')
#    f1 = alipy.metrics.f1_score(y_true=y[test_idx], y_pred=pred, labels=[-1,0,1], pos_label=1, average= None, sample_weight=None)
    saver.set_initial_point(test_auc)

    while len(unlab_ind.index)>0:

        # Select a subset of Uind according to the query strategy
        # Passing any sklearn models with proba_predict method are ok
        #select_ind = uncertainStrategy.select(label_ind, unlab_ind, model=model, batch_size=1)
        select_ind1, select_ind2 = selectInstance(label_ind, unlab_ind, model=model)
        # or pass your proba predict result
        # prob_pred = model.predict_proba(x[unlab_ind])
        # select_ind = uncertainStrategy.select_by_prediction_mat(unlabel_index=unlab_ind, predict=prob_pred, batch_size=1)
      #  print(select_ind1, select_ind2)
        
      #  print(select_ind1 in unlab_ind.index,select_ind2 in unlab_ind.index)
        
        
        if select_ind1 in unlab_ind.index:
          #  print(select_ind1 in unlab_ind.index)
            label_ind.update(select_ind1)
            unlab_ind.difference_update(select_ind1)
           # print(select_ind1 in unlab_ind.index)
        else:
            select_ind = uncertainStrategy.select(label_ind, unlab_ind, model=model, batch_size=1)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
        if select_ind2 in unlab_ind.index:
            #print(select_ind2 in unlab_ind.index)
            label_ind.update(select_ind2)
            unlab_ind.difference_update(select_ind2)
            #print(select_ind2 in unlab_ind.index)
        else:
            select_ind = uncertainStrategy.select(label_ind, unlab_ind, model=model, batch_size=1)
            label_ind.update(select_ind)
            unlab_ind.difference_update(select_ind)
        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        y_predict_prob = model.predict_proba(X[test_idx, :])
        test_auc = metrics.roc_auc_score(y[test_idx],y_predict_prob,multi_class='ovr')
       # accuracy = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=pred, performance_metric='f1_score')
    #    f1 = alipy.metrics.f1_score(y_true=y[test_idx], y_pred=pred, labels=[-1,0,1], pos_label=1, average= None, sample_weight=None)
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
df.to_excel('./4/math_lc_ans.xlsx')
print('./4/math_lc_ans.xlsx')
analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='QUIRE', method_results=unc_result)
print(analyser)
analyser.plot_learning_curves(title='Example of AL', std_area=True)