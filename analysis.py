import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import h5py

# for hierarchical
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

# for KMeans
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D

# for biplot
import seaborn as sns
import pandas as pd


def generate_cluster(data, method="DBSCAN", eps=1, n_class=1):
    """
    not sure if we want to normalize or standardize on the
    photometrics side though.
    DBSCAN implementation here.
    This function generate the cluster in each bin
    based on the photometrics feature.
    reference:
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    http://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
    """

    # visualization problem
    if method == "DBSCAN":
        dbscan_cluster = DBSCAN()
        db = DBSCAN(eps=0.01, min_samples=2).fit(data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        unique_labels = set(labels.flatten())
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = data[class_member_mask & core_samples_mask]
            plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()
        return labels

    if method == "hierarchical":
        Z = linkage(data, 'ward')
        # c, coph_dists = cophenet(Z, pdist(data))
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=12,  # show only the last p merged clusters
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,  # to get a distribution impression in truncated branches
        )
        plt.show()
        return Z

    if method == "ch_idx":
        distortions = []
        K = range(2, 10)
        for k in K:
            print(k)
            kmeanModel = KMeans(n_clusters=k).fit(data)
            kmeanModel = kmeanModel.fit(data)
            labels = kmeanModel.predict(data)
            distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

        # Plot the elbow
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    if method == "Kmeans":
        cmhot = plt.get_cmap("hot")
        kmeanModel = KMeans(n_clusters=n_class).fit(data)
        kmeanModel = kmeanModel.fit(data)
        labels = kmeanModel.predict(data)
        C = kmeanModel.cluster_centers_
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=labels, cmap=cmhot)
        return labels


# h5 = h5py.File("clusters_summary.hdf5", mode='w')
# h5 = h5.close()

def store_rst(preds, name='KMeans', file_name='clusters_summary.hdf5'):
    h5 = h5py.File(file_name, mode='r+')
    h5.create_dataset(name, data=preds)
    h5.close()


