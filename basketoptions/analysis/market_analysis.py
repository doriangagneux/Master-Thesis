import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data, scaler

def perform_pca(data, n_components=None):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    loadings = pca.components_
    return principal_components, explained_variance_ratio, loadings

def plot_explained_variance(explained_variance_ratio):
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(explained_variance_ratio))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Principal Components')
    plt.grid(True)
    plt.show()

def plot_principal_components(principal_components):
    plt.figure(figsize=(10, 5))
    plt.scatter(principal_components[:, 0], principal_components[:, 1])
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('First Two Principal Components')
    plt.grid(True)
    plt.show()