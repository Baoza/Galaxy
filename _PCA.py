import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# for PCA
# Pay attention to this
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def adaptive_PCA(data, n_pca=10):
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    df = pd.DataFrame(data)
    pca_output = PCA(n_components=n_pca).fit(data)
    X_pca = pca_output.transform(data)
    df_pca = pd.DataFrame(X_pca)
    df_pca.columns = ['pc'+str(i) for i in range(n_pca)]

    sns.lmplot('pc1', 'pc2', data=df_pca[0:1000], fit_reg=False,
               size=5, scatter_kws={"s": 100})

    # set the maximum variance of the first two PCs
    # this will be the end point of the arrow of each original features**
    xvector = pca_output.components_[0, 0:1000]
    yvector = pca_output.components_[1, 0:1000]

    # value of the first two PCs, set the x, y axis boundary
    xs = pca_output.transform(data)[:, 0][0:1000]
    ys = pca_output.transform(data)[:, 1][0:1000]

    # visualize projections

    # add col names
    # Note: scale values for arrows and text are a bit inelegant as of now,
    #       so feel free to play around with them
    # plot data

    for i in range(len(xvector)):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        # we can adjust length and the size of the arrow
        plt.arrow(0, 0, xvector[i] * max(xs), yvector[i] * max(ys),
                  color='r', width=0.005, head_width=0.05)
        plt.text(xvector[i] * max(xs) * 1.1, yvector[i] * max(ys) * 1.1,
                 list(df.columns.values)[i], color='r')

    # for i in range(len(xs)):
    #    plt.text(xs[i] * 1.08, ys[i] * 1.08, list(data.index)[i], color='b')  # index number of each observations
    plt.title('PCA Plot of first PCs')

    print("variance ratio explained",
          sum(pca_output.explained_variance_ratio_))

    return X_pca


def visua_PCA(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    result = result[0:5000]
    ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap="Set2_r", s=60)

    # make simple, bare axis lines through space:
    xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0, 0), (min(result['PCA2']), max(result['PCA2'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA on galaxy data set")
