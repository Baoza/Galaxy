from _bin import *
import re
from sklearn.cluster import KMeans
import KNN
from kmeans import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import h5py


def read_h5py(bin_idx=0):
    h5 = h5py.File("bin"+bin_idx+".hdf5", mode="r+")
    _, col_spec = h5["spec_data"].shape
    _, col_summary = h5["summary_data"].shape
    spec_data = h5["spec_data"][:, 0:col_spec].copy()
    summary_data = h5["summary_data"][:, 0:col_summary].copy()
    return spec_data, summary_data


# PCA
def adaptive_PCA(data, n_pca=10):
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    pca = PCA(n_components=n_pca).fit(data)
    trans = pca.transform(data)
    trans = trans[:, 1:] / trans[:, 0].reshape(-1, 1)
    print("variance ratio explained",
          sum(pca.explained_variance_ratio_))
    return trans

spec_data, summary_data = read_h5py(bin_idx=0)
spec_trans = adaptive_PCA(np.delete(spec_data, 1, 1))
summary_trans = adaptive_PCA(summary_data)

# k-means
centroids_spec, clusterAssment_spec = kmeans(spec_trans, 20)
centroids_summary, clusterAssment_summary = kmeans(summary_trans, 10)


"""
for i, em_line in enumerate(emission_lines):
    rst_list = []
    for map in missing_list:
        rst_list.append(map[em_line])
    fig, ax = plt.subplots()
    ax.plot(rst_list)
    ax.set_title(em_line)



# plot one of the emission line
em_line = 'Flux_HeII_3203'
rst_list = []
for map in zero_list:
    rst_list.append(map[em_line])
plt.plot(rst_list)
plt.title(em_line)


# PCA - 80 columns: 6-51 features; 52-79 components
idx0 = 2
temp1 = bin_data[idx0].copy()

temp1.replace(-9999, np.nan)
temp1.replace(0, np.nan)

"""



