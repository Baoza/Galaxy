import analysis
import h5py
import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np

from analysis import read_h5py
from add_flux_info import read_rst
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.spatial.distance import mahalanobis

spec_data, summary_data = read_h5py(0)
KMeans_, DB, centroids, core_sample_indices = read_rst("clusters_summary.hdf5")

a, b = summary_data.shape
data = np.zeros([a, b+1])
data[:, 0:b] = summary_data
data[:, b] = KMeans_
data = pd.DataFrame(data)

# to do: record the columns' name into h5py

# parallel_coordinates plot for clustering
# how each variable contributes
data.columns = [str(i) for i in range(46)]
plt.figure()
parallel_coordinates(data.iloc[1:1000], '45')

# biplot for clustering: based on parallel, chose No.5 No.44
temp = data[0:1000]
sns.lmplot('5', '44', data=temp, fit_reg=False, hue='45',
           size=15, scatter_kws={"s": 100})

"""
# set the maximum variance of the first two PCs
# this will be the end point of the arrow of each **original features**
xvector = temp['5']
yvector = temp['44']

# value of the first two PCs, set the x, y axis boundary
xs = temp['5']
ys = temp['44']

## visualize projections

## add col names
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them
for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
    # we can adjust length and the size of the arrow
    plt.arrow(0, 0, xvector[i] * max(xs), yvector[i] * max(ys),
              color='r', width=0.005, head_width=0.05)
    plt.text(xvector[i] * max(xs) * 1.1, yvector[i] * max(ys) * 1.1,
             list(data.columns.values)[i], color='r')

# for i in range(len(xs)):
#    plt.text(xs[i] * 1.08, ys[i] * 1.08, list(data.index)[i], color='b')  # index number of each observations
plt.title('biplots for clustering')

"""


# Mahalanobis distance
# pay attention to the negative value
def _cal_mahalanobis(xx, yy):
    X = np.vstack([xx, yy])
    V = np.cov(X.T)
    p = np.linalg.inv(V)
    D = mahalanobis(xx, yy, p)
    if D > 0:
        return D
    else:
        return -1


# calculate for one cluster
def cal_mahalanobis(data, centroids, cluster_idx = 0):
    class_member_mask = (data['45'] == cluster_idx)
    temp = data[class_member_mask]
    temp = temp.iloc[:, 0:45]
    c = centroids[cluster_idx]
    distance = []
    for i in range(temp.shape[0]):
        record = temp.iloc[i] + 0.000001
        distance.append(_cal_mahalanobis(record, c))
    distance = np.asarray(distance)
    nonzero_list = np.nonzero(distance+1)
    distance = distance[nonzero_list]

    fig, ax = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax.hist(distance, 100, density=True, histtype='step', label='Empirical')


