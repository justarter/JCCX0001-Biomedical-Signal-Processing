from sklearn import svm
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc, f1_score

#加载数据集
def loadDataSet(filename):
    dataSet = pd.read_csv(filename, header=None)
    dataSet = dataSet.fillna(0)# Nan -> 0
    dataSet = dataSet.values
    m, n = dataSet.shape
    data_X = dataSet[:, :n-1]
    data_Y = np.where(dataSet[:, n - 1]>0, 1,-1)

    return data_X, data_Y

train_data_filename = "gpl96.csv"
test_data_filename = "gpl97.csv"
X_train, Y_train = loadDataSet(train_data_filename)
X_test, Y_test = loadDataSet(test_data_filename)
clf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovo')
clf.fit(X_train, Y_train)

train_accuracy = clf.score(X_train, Y_train)
print("train_accuracy:", train_accuracy)

test_accuracy = clf.score(X_test, Y_test)
print("test_accuracy:", test_accuracy)

# print('train_decision_function:',clf.decision_function(X_train))
# print('predict_result:', clf.predict(X_train))
# print(clf.predict(X_test), Y_test)
f1 = f1_score(Y_test, clf.predict(X_test))
print('F1 score', f1)

test_predict_label = clf.decision_function(X_test)
fpr, tpr, threshold = roc_curve(Y_test, test_predict_label)

# print(fpr,tpr,threshold)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()


