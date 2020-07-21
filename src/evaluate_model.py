import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, confusion_matrix

"""
Basic functions to quickly evaluate models
"""


def analyse_result(y_pred, y_test, model = 'model'):
    """
    Descriptive metrics on model performance in tabular format.

    Parameters:
    clf: sklearn model for which we want to estimate performance
    X_test: test input matrix
    y_test: test output labels
    model (str): model-name as column-name in the table
                 (usefull when merging for comparisons)

    Result:
    Dataframe with metrics as rows
    """
    if type(y_pred) == list:
        clf = y_pred[0]
        X_test = y_pred[1]

        # predict on X_test
        y_pred = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    analytics = {'accuracy' : accuracy_score(y_test, y_pred),
                 'recall' : recall_score(y_test, y_pred),
                 'precision' : precision_score(y_test, y_pred),
                 'specificity' : tn / (tn+fp),
                 'f1-score' : f1_score(y_test, y_pred),
                 'AUC score': roc_auc_score(y_test, y_pred)}

    return pd.DataFrame.from_dict(analytics, orient='index', columns = [model])


def feature_importance(X_train, clf):
    """
    Feature importance as table with the correct parameter labels

    Parameters:
    X_train: training input dataframe (used for labels only)
    clf: sklearn model for which we want to estimate performance

    Result:
    f_imp: pandas dataframe with the parameters and their
           relative importance as rows
    """
    f_imp = pd.DataFrame([list(X_train.columns),clf.feature_importances_]).T
    f_imp.columns = ['feature', 'importance']
    f_imp = f_imp[~f_imp['importance'].isnull()].sort_values('importance', ascending=False).reset_index(drop=True)
    return f_imp


def feature_importance_plot(f_imp, cutoff = 0.005, figsize = (8,8)):
    """
    Barplot of relative feature importance for top features

    Parameters:
    f_imp: feature importance dataframe (see feature_importance)
    cutoff: select features where relative importance > cutoff (range 0 - 1)
    f_size: matplotlib figsize attribute (width,height)
    """
    plt.figure(figsize = figsize)
    subset = f_imp[f_imp['importance'] > cutoff]
    plt.barh(data = subset, y = 'feature', width = 'importance')
    plt.title('explained by these features: ' + str(subset.importance.sum()) )
    plt.show()


def roc_curve_plot(clfs, X_test, y_test, labels):
    """
    Plot the models ROC curve.
    NB: alls models in clfs should take the same number of features (from X_test)

    Parameters:
    clfs: list of sklearn models for which we want to estimate performance
    X_test: test input matrix
    y_test: test output labels
    labels: name for the models to be used in the plot
    """
    # calculate ROC curves for all models in clfs
    for i, clf in enumerate(clfs):
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label = labels[i])

    # plot naive ROC curve (diagonal)
    plt.plot([0,1],[0,1], linestyle = '--', c = 'grey', alpha = 0.3)

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
