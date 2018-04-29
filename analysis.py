from numpy import *
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

# Pay attention to this
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def read_h5py(bin_idx=0):
    h5 = h5py.File("bin"+str(bin_idx)+".hdf5", mode="r+")
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


# make PCA plots
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


def generate_cluster(data, method="DBSCAN"):
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
        db = DBSCAN(eps=0.3, min_samples=3).fit(data)
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
            plt.plot(xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
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

    if method == "Kmeans":
        # centroids, clusterAssement = kmeans(data, 50)
        n_clusters = 50
        estimator = KMeans(n_clusters)
        estimator.fit(data)
        label_pred = estimator.labels_
        centroids = estimator.cluster_centers_
        inertia = estimator.inertia_

        unique_labels = set(label_pred.flatten())
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (label_pred == k)

            xy = data[class_member_mask]
            plt.plot(xy[:, 3], xy[:, 4], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=4)

            plt.plot(centroids[:, 3], centroids[:, 4], 'o', markerfacecolor=tuple([0, 0, 0, 1]),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters)
        plt.show()


