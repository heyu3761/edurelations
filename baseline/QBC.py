# -*- coding: utf-8 -*-
from alipy import ToolBox
import copy
import numpy as np
import  pandas  as pd
from sklearn.ensemble import GradientBoostingClassifier
import alipy
from sklearn import metrics
from  alipy.query_strategy import QueryInstanceQBC
from pandas import DataFrame

df=pd.read_excel('../cs.xlsx')
dataset = df.values

# 补充0
dataset[np.isnan(dataset)] = 0
print(dataset.shape)

feature = dataset[:,2:20]
label = dataset[:,20]

X, y = feature, label
alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='./2')

# Split data
alibox.split_AL(test_ratio=0.2, initial_label_rate=0.1, split_count=1)

# Use the default Logistic Regression classifier
#model = alibox.get_default_model()
model = GradientBoostingClassifier(n_estimators=40,learning_rate=0.1,min_samples_split=700,min_samples_leaf=40,max_depth=7,max_features='sqrt')

# The cost budget is 50 times querying
#stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 100000)

# Use pre-defined strategy
uncertainStrategy = QueryInstanceQBC(X=X, y=y, method='query_by_bagging', disagreement='vote_entropy')
unc_result = []
auc_ret = []
print("model=gbdt")
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
        select_ind = uncertainStrategy.select(label_ind, unlab_ind, model=model, batch_size=2)
        # or pass your proba predict result
        # prob_pred = model.predict_proba(x[unlab_ind])
        # select_ind = uncertainStrategy.select_by_prediction_mat(unlabel_index=unlab_ind, predict=prob_pred, batch_size=1)

        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        y_predict_prob = model.predict_proba(X[test_idx, :])
        test_auc = metrics.roc_auc_score(y[test_idx],y_predict_prob,multi_class='ovr')
        auc_ret.append(copy.deepcopy(test_auc))
       # accuracy = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=pred, performance_metric='f1_score')
    #    f1 = alipy.metrics.f1_score(y_true=y[test_idx], y_pred=pred, labels=[-1,0,1], pos_label=1, average= None, sample_weight=None)

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind, performance=test_auc)
        saver.add_state(st)
       # saver.save()

        # Passing the current progress to stopping criterion object
    #    stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
#    stopping_criterion.reset()
    unc_result.append(copy.deepcopy(saver))
df=DataFrame(auc_ret)
df.to_excel('./2/cs_withmodel_ans2.xlsx')
print('./2/cs_withmodel_ans2.xlsx')
analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='uncertainty', method_results=unc_result)
print(analyser)
analyser.plot_learning_curves(title='Example of AL', std_area=True)