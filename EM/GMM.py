import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import time
multivariate_normal_pdf = multivariate_normal.pdf

def load_data(PATH="GMM_EM_data_for_clustering.csv"):
    raw_data = np.array(pd.read_csv(PATH))
    data = raw_data[:, 1:3]
    label = raw_data[:, -1:].reshape(-1,).astype(np.int32)# K =4
    # print(data.shape, label.shape) #(1500*3), (1500,)
    return data, label

def init_random(data, K):# random choose K centroids
    index = np.random.choice(range(data.shape[0]), K)
    centroids = data[index, :]
    return centroids

def initialize(data, centroids):
    K = centroids.shape[0]
    means = centroids #K*2
    alphas = np.ones(K) / K
    variances = np.array([np.cov(data.T)]*K)# K*2*2
    return means, variances, alphas

def E_step(data, means, variances, alphas):
    N, K = data.shape[0], means.shape[0]
    gammas = np.zeros((N, K))
    for k in range(K):
        gammas[:, k] = alphas[k]*multivariate_normal_pdf(data, mean=means[k], cov=variances[k])
    gammas /= np.sum(gammas, axis=1)[:, np.newaxis]
    # print(gammas.shape)# N * K
    return gammas

def M_step(data, means, gammas):
    N, K = gammas.shape
    dimension = data.shape[1]
    new_means = np.dot(gammas.T, data) / np.sum(gammas, axis=0)[:, np.newaxis]
    # print("new means", new_means.shape)
    new_variances = np.zeros((K, dimension, dimension))
    for k in range(K):
        gam = gammas[:, k][:, np.newaxis]
        # print("gam",gam.shape)
        tmp = np.dot((data - means[k]).T, (data - means[k])*gam)
        new_variances[k] = tmp / np.sum(gammas, axis=0)[k]
    new_alphas = np.sum(gammas, axis=0) / N
    return new_means, new_variances, new_alphas

def Gaussian_Mixture(data, centroids, maxIteration):
    means, variances, alphas = initialize(data, centroids)
    for iter in range(maxIteration):
        print("Iteration {} begin".format(iter))
        # print(alphas.sum())
        gammas = E_step(data, means, variances, alphas)
        assignments = np.argmax(gammas, axis=1)
        # print(assignments.shape)
        means, variances, alphas = M_step(data, means, gammas)
    return assignments

def plot_clusters(X, label):
    _, ax = plt.subplots()
    plt.figure(figsize=(10, 8))
    ax.scatter(X[:, 0], X[:, 1], s=5, c=label)
    plt.show()

data, label = load_data()
K = np.max(label) + 1# K = 4
MaxIter = 100

beg = time.time()
centroids = init_random(data, K)
assignments = Gaussian_Mixture(data, centroids, MaxIter)
# print(np.unique(assignments))
time1 = time.time()-beg

beg = time.time()
gmm = GaussianMixture(n_components=K, init_params='random', max_iter=MaxIter).fit(data)
y_gmm = gmm.predict(data)
time2 = time.time()-beg

beg = time.time()
gmm = GaussianMixture(n_components=K, init_params='kmeans', max_iter=MaxIter).fit(data)
y_gmm2 = gmm.predict(data)
time3 = time.time()-beg

plt.figure()
plt.subplot(221)
plt.scatter(data[:, 0], data[:, 1], s=5, c=label)
plt.title('GroundTruth')
plt.subplot(222)
plt.scatter(data[:, 0], data[:, 1], s=5, c=assignments)
plt.title('My GMM Clustering')
plt.text(-10, -10, '{:.3f}s'.format(time1))

plt.subplot(223)
plt.scatter(data[:, 0], data[:, 1], s=5, c=y_gmm)
plt.title('Sklearn GMM Clustering (Random)')
plt.text(-10, -10, '{:.3f}s'.format(time2))
plt.subplot(224)
plt.scatter(data[:, 0], data[:, 1], s=5, c=y_gmm2)
plt.title('Sklearn GMM Clustering (Kmeans)')
plt.text(-10, -10, '{:.3f}s'.format(time3))
plt.suptitle("The Number of Iteration: {}".format(MaxIter))
plt.show()
