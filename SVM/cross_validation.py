from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score

#加载数据集
def loadDataSet(filename):
    dataSet = pd.read_csv(filename, header=None)
    dataSet = dataSet.fillna(0)# Nan -> 0
    dataSet = dataSet.values
    m, n = dataSet.shape
    data_X = dataSet[:, :n-1]
    data_Y = np.where(dataSet[:, n - 1] > 0, 1, -1)

    return data_X, data_Y

def loadDataSet_logis(filename):
    dataSet = pd.read_csv(filename, header=None)
    dataSet = dataSet.fillna(0)  # Nan -> 0
    dataSet = dataSet.values
    m, n = dataSet.shape
    data_X = dataSet[:, :n - 1]
    data_Y = np.where(dataSet[:, n - 1] > 0, 1, 0)

    return data_X, data_Y

train_data_filename = "gpl96.csv"
X_train, Y_train = loadDataSet(train_data_filename)

clf = svm.SVC(kernel='linear', C=0.8)

print("SVM accuracy:", cross_val_score(clf, X_train, Y_train, cv=5, scoring='accuracy').mean())
print('SVM F1 score:', cross_val_score(clf, X_train, Y_train, cv=5, scoring='f1').mean())
print('SVM AUC', cross_val_score(clf, X_train, Y_train, cv=5, scoring='roc_auc').mean())


X_train, Y_train = loadDataSet_logis(train_data_filename)

logis_clf = LogisticRegression(solver='liblinear')

print("Logis accuracy:", cross_val_score(logis_clf, X_train, Y_train, cv=5, scoring='accuracy').mean())
print('Logis F1 score:', cross_val_score(logis_clf, X_train, Y_train, cv=5, scoring='f1').mean())
print('Logis AUC', cross_val_score(logis_clf, X_train, Y_train, cv=5, scoring='roc_auc').mean())







# fpr, tpr, _ = roc_curve(Y_train, Y_predicted)#这里不对
# roc_auc = auc(fpr, tpr)

# plt.figure()
# lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve')
# plt.legend(loc="lower right")
# plt.show()

# X_train, Y_train = loadDataSet_logis(train_data_filename)
#
# logis_clf = LogisticRegression(solver='liblinear')
# logis_Y_predicted = cross_val_score(logis_clf, X_train, Y_train, cv=5)
# print("Logis accuracy:", accuracy_score(Y_train, logis_Y_predicted))
# print('Logis F1 score:', f1_score(Y_train,  logis_Y_predicted))
#
# fpr, tpr, _ = roc_curve(Y_train, logis_Y_predicted)#这里不对
# roc_auc = auc(fpr, tpr)
# print('Logis AUC', roc_auc)
# plt.figure()
# lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve')
# plt.legend(loc="lower right")
# plt.show()

