import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import time

def normalize(X):
    # mean value of each column
    mu = np.mean(X, axis=0)
    X = X - mu
    # std of each column
    std = np.std(X, axis=0)
    std_copy = std.copy()
    std_copy[std == 0] = 1
    X_bar = X / std_copy
    return X_bar, mu, std

def compute_eig(S):
    eigen_values, eigen_vectors = np.linalg.eigh(S)
    # increase order
    sorted_eigen = np.argsort(-eigen_values)
    eigen_values = eigen_values[sorted_eigen]
    eigen_vectors = eigen_vectors[:, sorted_eigen]
    return (eigen_values, eigen_vectors)

'''
mse by myself(also could use sklearn's mean_squared_error)
'''
def mse(truth, predicted):
    return np.square(truth - predicted).sum(axis=1).mean()

'''
PCA for low dimensional data
'''
def PCA_low(X, num_components):
    S = np.cov(X.T)#X^T X, 和公式意思是一样的但是写法反了一下,D*D
    # print(S.shape)
    eigen_values, eigen_vectors = compute_eig(S)
    # print(eigen_values.shape, eigen_vectors.shape)
    U = eigen_vectors[:, range(num_components)]#D*K
    # print(U.shape)
    lowData = np.dot(X, U)# N*K
    X_reconstruct = np.dot(lowData, U.T)

    return X_reconstruct

def apply_PCA_low(num_components, num_datapoints): #num_datapoints > 784
    raw_data = loadmat('mnist-original.mat')
    # print(raw_data['data'], raw_data['label'])
    data = raw_data['data'] #784*70000
    # print(data.shape)
    X = data.T[range(num_datapoints), :]/255 #N*D

    # pic = X[2, :] * 255
    # pic = pic.reshape(28, 28)
    # plt.imshow(pic, cmap='gray')
    # plt.show()

    X_bar, mu, std = normalize(X)
    X_reconstruct = PCA_low(X_bar, num_components)
    # loss = mse(X_bar, X_reconstruct)
    loss = mean_squared_error(X_bar, X_reconstruct)

    X_reconstruct = (X_reconstruct * std) + mu
    print("PCA-low loss: ", loss)

    # pic = X_reconstruct[2, :] * 255
    # pic = pic.reshape(28, 28)
    # plt.imshow(pic, cmap='gray')
    # plt.show()
    return loss

# apply_PCA_low(378, 850)
def change_num_components_PCA_low():
    res = {}
    for i in range(350, 400):
        print("Num of components: ", i)
        res[i] = apply_PCA_low(i, 850)

    plt.plot(res.keys(), res.values())
    plt.xlabel("Num of components")
    plt.ylabel("MSE loss")
    plt.title("PCA-low with 850 samples")
    plt.show()

# change_num_components_PCA_low()


def change_num_datapoints_PCA_low():
    res = {}
    for i in range(785, 885):
        print("Num of samples: ", i)
        res[i] = apply_PCA_low(380, i)

    plt.plot(res.keys(), res.values())
    plt.xlabel("Num of samples")
    plt.ylabel("MSE loss")
    plt.title("PCA-low with 380 components")
    plt.show()

# change_num_datapoints_PCA_low()

'''
PCA for high dimensional data 
'''
def PCA_high(X, num_components):
    S = np.cov(X)  # X X^T, N * N
    num_datapoints, _ = X.shape
    # print(S.shape)
    eigen_values, eigen_vectors = compute_eig(S)
    U = eigen_vectors[:, range(num_components)]  # N*K
    # print(np.sum(np.square(U[:,3])))

    # 这里归一化我直接除以模长（公式里说除以 sqrt（N*lambda），但这样还是没有归一化啊？）
    # actual_U = np.dot(X.T, U)/(np.sqrt(num_datapoints * eigen_values[range(num_components)]))#D*K
    actual_U = np.dot(X.T, U)
    # print(np.sqrt(num_datapoints * eigen_values[range(num_components)]).shape)
    actual_U /= np.linalg.norm(actual_U, axis=0)
    # print(np.sum(np.square(actual_U[:, 0])))

    lowData = np.dot(X, actual_U)  # N*K
    X_reconstruct = np.dot(lowData, actual_U.T)
    return X_reconstruct

def apply_PCA_high(num_components, num_datapoints):#num_datapoints < 784
    raw_data = loadmat('mnist-original.mat')
    data = raw_data['data'] #784*70000
    X = data.T[range(num_datapoints), :]/255 #N*D

    # pic = X[0, :] * 255
    # pic = pic.reshape(28, 28)
    # plt.imshow(pic, cmap='gray')
    # plt.show()

    X_bar, mu, std = normalize(X)
    X_reconstruct = PCA_high(X_bar, num_components)
    loss = mean_squared_error(X_bar, X_reconstruct)
    # loss = mse(X_bar, X_reconstruct)
    X_reconstruct = (X_reconstruct * std) + mu

    print("PCA-high loss: ", loss)
    # pic = X_reconstruct[0, :] * 255
    # pic = pic.reshape(28, 28)
    # plt.imshow(pic, cmap='gray')
    # plt.show()
    return loss

# apply_PCA_low(20, 250)
# apply_PCA_high(20, 250)

def change_num_components_PCA_high():
    res = {}
    for i in range(250, 300):
        print("Num of components: ", i)
        res[i] = apply_PCA_high(i, 500)

    plt.plot(res.keys(), res.values())
    plt.xlabel("Num of components")
    plt.ylabel("MSE loss")
    plt.title("PCA-high with 500 samples")
    plt.show()

# change_num_components_PCA_high()

def change_num_datapoints_PCA_high():
    res = {}
    for i in range(400, 600):
        print("Num of samples: ", i)
        res[i] = apply_PCA_high(80, i)

    plt.plot(res.keys(), res.values())
    plt.xlabel("Num of samples")
    plt.ylabel("MSE loss")
    plt.title("PCA-high with 80 components")
    plt.show()

change_num_datapoints_PCA_high()

'''
Sklearn PCA
'''
def plot_eigenvec_ratio(importance):
    length = len(importance)
    plt.scatter(range(1, length+1), importance)
    plt.plot(range(1, length+1), importance)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

def apply_sklearn_PCA(num_components, num_datapoints):
    raw_data = loadmat('mnist-original.mat')
    data = raw_data['data'] #784*70000
    X = data.T[range(num_datapoints), :]/255 #num_datapoints*784
    X_bar, mu, std = normalize(X)
    pca = PCA(n_components=num_components)
    lowData = pca.fit_transform(X_bar)
    # plot_eigenvec_ratio(pca.explained_variance_ratio_)
    X_reconstruct = pca.inverse_transform(lowData)
    loss = mean_squared_error(X_bar, X_reconstruct)

    X_reconstruct = (X_reconstruct * std) + mu

    print("PCA-sklearn loss: ", loss)
    return loss

# def change_num_datapoints_PCA_sklearn():# for test
#     res = {}
#     for i in range(600, 650):
#         print("Num of samples: ", i)
#         res[i] = apply_sklearn_PCA(350, i)
#
#     plt.plot(res.keys(), res.values())
#     plt.xlabel("Num of samples")
#     plt.ylabel("MSE loss")
#     plt.title("PCA-sklearn with 380 components")
#     plt.show()

'''
This part is for comparing
'''
def PCA_low_time(n):
    beg = time.time()
    for i in range(50):
        apply_PCA_low(n, 200)
    return time.time()-beg

def PCA_high_time(n):
    beg = time.time()
    for i in range(50):
        apply_PCA_high(n, 200)
    return time.time()-beg

def compare_time():
    low_res = []
    high_res = []
    for n in range(10, 50):
        low_res.append(PCA_low_time(n))
        high_res.append(PCA_high_time(n))
    plt.plot(range(10, 50), low_res, color='green', label='PCA-low')
    plt.plot(range(10, 50), high_res, color='red', label='PCA-high')
    plt.legend()
    plt.xlabel("Num of components")
    plt.ylabel("Running time(50 times)")
    plt.title("Fix Num of datapoints=200")
    plt.show()

# compare_time()

def compare_three():
    # x = []
    # y1 = []
    # y2 = []
    # y3 = []
    # for i in range(10, 100, 10):
    #     x.append(i)
    #     y1.append(apply_PCA_low(i, 600))
    #     y2.append(apply_PCA_high(i, 600))
    #     y3.append(apply_sklearn_PCA(i, 600))
    # plt.plot(x, y1, color='green', label='PCA-low')
    # plt.plot(x, y2, color='red', label='PCA-high')
    # plt.plot(x, y3, color='skyblue', label='PCA-sklearn')
    # plt.legend()
    # plt.xlabel("Num of components")
    # plt.ylabel("MSE loss")
    # plt.title("Three PCAs while fixing num of samples=600")
    # plt.show()

    # x = []
    # y1 = []
    # y2 = []
    # y3 = []
    # for i in range(10, 100, 10):
    #     x.append(i)
    #     y1.append(apply_PCA_low(i, 800))
    #     y2.append(apply_PCA_high(i, 800))
    #     y3.append(apply_sklearn_PCA(i, 800))
    # plt.plot(x, y1, color='green', label='PCA-low')
    # plt.plot(x, y2, color='red', label='PCA-high')
    # plt.plot(x, y3, color='skyblue', label='PCA-sklearn')
    # plt.legend()
    # plt.xlabel("Num of components")
    # plt.ylabel("MSE loss")
    # plt.title("Three PCAs while fixing num of samples=800")
    # plt.show()

    x = []
    y1 = []
    y2 = []
    y3 = []
    for i in range(450, 950, 50):
        x.append(i)
        y1.append(apply_PCA_low(30, i))
        y2.append(apply_PCA_high(30, i))
        y3.append(apply_sklearn_PCA(30, i))

    plt.plot(x, y1, color='green', label='PCA-low')
    plt.plot(x, y2, color='red', label='PCA-high')
    plt.plot(x, y3, color='skyblue', label='PCA-sklearn')
    plt.legend()
    plt.xlabel("Num of datapoints")
    plt.ylabel("MSE loss")
    plt.title("Three PCAs while fixing num of components=30")
    plt.show()
# compare_three()
