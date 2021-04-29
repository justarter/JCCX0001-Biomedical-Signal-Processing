import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve,auc,f1_score,accuracy_score

'''
The codes are mainly from https://blog.csdn.net/troysps/article/details/80500253
'''

# 加载数据集
def loadDataSet(filename):
    dataSet = pd.read_csv(filename, header=None)
    dataSet = dataSet.fillna(0)  # Nan -> 0
    dataSet = dataSet.values
    m, n = dataSet.shape

    data_X = dataSet[:, :n - 1]
    data_Y = np.where(dataSet[:, n - 1] > 0, 1, -1)  # should change label 0 to -1
    return data_X, data_Y

class optStruct(object):
    #store all important values
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        """
        C: regularization parameter
        toler: tolerance
        kTup: info of kernel func
        """
        self.X = dataMatIn
        self.labelMat = classLabels
        self.m = np.shape(dataMatIn)[0]

        self.C = C
        self.tol = toler

        # init alphas and b
        self.alphas = np.mat(np.zeros((self.m , 1)))
        self.b = 0

        # record E, first column is valid flag, second column is E value.
        self.eCache = np.mat(np.zeros((self.m, 2)))

        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def kernelTrans(X, A, kTup):
    """
    X: training data
    A: X[i,:]
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T # m*n * n*1 = m*1
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('This Kernel is not implemented')
    return K

def calcEk(oS, i):
    """
    Ek=predicted value - truth
    """
    # predictEk = float(np.multiply(oS.alphas, oS.classLabels).T * (oS.X * oS.X[i, :].T)) + oS.b
    # multiply(mx1, mx1)= mx1 (mx1).T * (m,1) = 1x1
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, i] + oS.b)
    Ek = fXk - float(oS.labelMat[i])
    return Ek

def selectJrand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j

def selectJ(i, oS, Ei):
    """
    choose j s.t. maximize delta E
    record E
    """
    maxK = -1  # maxDeltaE's corresponding index
    maxDeltaE = 0
    Ej = 0

    oS.eCache[i] = [i, Ei]

    # 非零E值的行(index)的list列表, 所对应的alpha值
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 有效的缓存值列表
    # print('validEcacheList:', validEcacheList)
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:  # don't calc for i
                continue
            Ek = calcEk(oS, k)
            deltaE = np.abs(Ei - Ek)

            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:# if iterate for first time, random choose j
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)

    return j, Ej

def clipAlpha(aj, H, L):
    # L <= a <= H
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    """
    if (alphaPairChanged):return 1
    else :return 0
    """
    # compute E
    Ei = calcEk(oS, i)

    # KKT condition
    # yi*f(i) >= 1 and alpha = 0 (outside the boundary)
    # yi*f(i) == 1 and 0<alpha< C (on the boundary)
    # yi*f(i) <= 1 and alpha = C (between the boundary)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] < 0)):
        j, Ej = selectJ(i, oS, Ei)

        # record for later use
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # compute L and H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            # print("L==H")
            return 0

        # calc eta
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0:
            print('eta >= 0')
            return 0

        # calc new alpha_j and clip
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)

        # update E_j
        updateEk(oS, j)

        # if alpha_j change a little, return 0
        if (np.abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0

        # update new alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # update E_i
        updateEk(oS, i)

        # compute new b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('linear', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True  #=false if traverse all data once
    alphaPairsChanged = 0

    # 循环遍历条件: 循环maxIter次并且(alphaPairsChanged存在可改变的值) or 将所有数据遍历一次
    while (((iter < maxIter) and (alphaPairsChanged > 0 )) or (entireSet)):
        alphaPairsChanged = 0

        # entireSet=True or 非边界alpha对没有了, 就开始寻找alpha对,然后决定是否要进行else
        if entireSet:
            for i in range(oS.m):
                # 是否存在alpha对, 存在就+1
                alphaPairsChanged += innerL(i, oS)
                # print(alphaPairsChanged)
            iter += 1
        # 对以及存在的alpha对 选出非边界的alpha值 进行优化
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1

        # 如果找到alpha对 就优化非边界alpha值 否则继续寻找
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True

    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

C = 0.8
toler = 0.001
maxIter = 15

train_X, train_Y = loadDataSet("gpl96.csv")

b, alphas = smoP(train_X, train_Y, C, toler, maxIter, ("rbf", 0.5))
ws = calcWs(alphas, train_X, train_Y)  # 含有随机操作，所以有多种可能性结果
# print(b)
# print(np.dot(train_X, ws) + b)

train_Y_hat = np.where(np.dot(train_X, ws) + b > 0, 1, -1).reshape(-1, )
train_accuracy = accuracy_score(train_Y, train_Y_hat)
print("train accuracy:", train_accuracy)

test_X, test_Y = loadDataSet("gpl97.csv")
test_Y_hat = np.where(np.dot(test_X, ws) + b > 0, 1, -1).reshape(-1, )
test_accuracy = accuracy_score(test_Y, test_Y_hat)
print("test accuracy:", test_accuracy)

f1 = f1_score(test_Y, test_Y_hat)
print('F1 score', f1)

fpr, tpr, threshold = roc_curve(test_Y, np.dot(test_X, ws) + b)
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
