import analysis
import h5py
from pandas.plotting import parallel_coordinates
from analysis import read_h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# file_name = clusters_summary or clusters_flux
def read_rst(file_name='clusters_summary.hdf5'):
    h5 = h5py.File(file_name, mode='r')
    a = h5["KMeans"].shape
    KMeans_ = h5["KMeans"][0:a[0]].copy()
    DB = h5["DBSCAN"][0:a[0]].copy()
    a, b = h5["centroids"].shape
    centroids = h5["centroids"][0:a, 0:b].copy()
    a = h5["core_sample_indices"].shape
    core_sample_indices = h5["core_sample_indices"][0:a[0]].copy()
    h5.close()
    return KMeans_, DB, centroids, core_sample_indices


spec_data, summary_data = read_h5py(0)
KMeans_, DB, centroids, core_sample_indices = read_rst("clusters_summary.hdf5")

a, b = spec_data.shape
data = np.zeros([a, b+1])
data[:, 0:b] = spec_data
data[:, b] = KMeans_
data = pd.DataFrame(data)

# to do: record the columns' name into h5py

# parallel_coordinates plot for clustering
# how each variable contributes
data.columns = [str(i) for i in range(b+1)]


# KNN by using flux in one cluster
def KNN_within_cluster(KMeans_, spec_data, cluster_idx=0, k=3):
    class_member_mask = (data['16'] == cluster_idx)
    temp = data[class_member_mask]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(temp)
    distances, indices = nbrs.kneighbors(temp)
    return distances, indices


