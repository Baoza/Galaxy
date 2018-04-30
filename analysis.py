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

# for PCA
# Pay attention to this
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# for biplot
import seaborn as sns
import pandas as pd


def read_h5py(bin_idx=0):
    h5 = h5py.File("bin"+str(bin_idx)+".hdf5", mode="r+")
    _, col_spec = h5["spec_data"].shape
    _, col_summary = h5["summary_data"].shape
    spec_data = h5["spec_data"][:, 0:col_spec].copy()
    summary_data = h5["summary_data"][:, 0:col_summary].copy()
    h5.close()
    return spec_data, summary_data


# PCA and biplots
def adaptive_PCA(data, n_pca=10):
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    df = pd.DataFrame(data[0:1000])
    pca = PCA(n_components=n_pca).fit(data)
    X_pca = pca.transform(data)
    df_pca = pd.DataFrame(X_pca)
    df_pca.columns = ['pc'+str(i) for i in range(n_pca)]

    sns.lmplot('pc1', 'pc2', data=df_pca[0:1000], fit_reg=False,
               size=15, scatter_kws={"s": 100})

    # set the maximum variance of the first two PCs
    # this will be the end point of the arrow of each **original features**
    xvector = pca.components_[0, 0:1000]
    yvector = pca.components_[1, 0:1000]

    # value of the first two PCs, set the x, y axis boundary
    xs = pca.transform(data)[:, 0][0:1000]
    ys = pca.transform(data)[:, 1][0:1000]

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
                 list(df.columns.values)[i], color='r')

    #for i in range(len(xs)):
    #    plt.text(xs[i] * 1.08, ys[i] * 1.08, list(data.index)[i], color='b')  # index number of each observations
    plt.title('PCA Plot of first PCs')

    print("variance ratio explained",
          sum(pca.explained_variance_ratio_))
    return X_pca


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
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
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
        estimator = KMeans(n_clusters=n_clusters)
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

            #xy = data[class_member_mask]
            #plt.plot(xy[:, 3], xy[:, 4], 'o', markerfacecolor=tuple(col),
            #         markeredgecolor='k', markersize=4)

            plt.plot(centroids[:, 0], centroids[:, 1], 'o', markerfacecolor=tuple([0, 0, 0, 1]),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters)
        plt.show()


# h5 = h5py.File("clusters_summary.hdf5", mode='w')
# h5 = h5.close()

def store_rst(preds, name='KMeans', file_name='clusters_summary.hdf5'):
    h5 = h5py.File(file_name, mode='r+')
    h5.create_dataset(name, data=preds)
    h5.close()


