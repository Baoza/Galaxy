import numpy as np
import analysis
import data_processing
from sklearn.neighbors import NearestNeighbors
import _PCA
import re
import pandas as pd
import seaborn as sns
import h5py

# emission_lines list
emission_cols = ['Flux_HeII_3203', 'Flux_NeV_3345', 'Flux_NeV_3425',
                 'Flux_OII_3726', 'Flux_OII_3728', 'Flux_NeIII_3868',
                 'Flux_NeIII_3967', 'Flux_H5_3889', 'Flux_He_3970',
                 'Flux_Hd_4101', 'Flux_Hg_4340', 'Flux_OIII_4363',
                 'Flux_HeII_4685', 'Flux_ArIV_4711', 'Flux_ArIV_4740',
                 'Flux_Hb_4861', 'Flux_OIII_4958', 'Flux_OIII_5006',
                 'Flux_NI_5197', 'Flux_NI_5200', 'Flux_HeI_5875',
                 'Flux_OI_6300', 'Flux_OI_6363', 'Flux_NII_6547',
                 'Flux_Ha_6562', 'Flux_NII_6583', 'Flux_SII_6716',
                 'Flux_SII_6730']


class bin_analysis():
    """
    analysis galaxy properties within a bin
    """
    def __init__(self, emission_cols, bin=0):
        self.bin = bin
        self.data = data_processing.read_h5py(self.bin)
        self.spec, self.summary = self.sep_line(emission_cols)

    def sep_line(self, emission_line):
        cols = [column for column in self.data.columns \
                if not re.match(r'.*[Ee]rr.*', column)]

        summary_cols = [col for col in cols if col not in emission_cols]
        summary_cols = summary_cols[2:len(summary_cols)]
        spec_cols = [col for col in cols if col not in summary_cols and col in emission_cols]
        return spec_cols, summary_cols

    def summary_PCA(self, components=3):
        summary_pca = _PCA.adaptive_PCA(self.data[self.summary[0:4]], components)
        trans_summary = summary_pca.transform(bin.data[bin.summary[0:4]])
        trans_summary = pd.DataFrame(trans_summary,
                                  columns=['PCA%i' % i for i in range(components)])
        _PCA.visua_PCA(trans_summary)
        return trans_summary

    def ch_idx(self, data):
        return analysis.generate_cluster(data, method="ch_idx")

    def kmeans(self, data, n_class=4):
        return analysis.generate_cluster(data, method="Kmeans")

    def KNN(self, data, n_neighbours):
        nbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
        return distances, indices

    def clustering(self, data, eps):
        return analysis.generate_cluster(data, eps=eps)


if __name__ == '__main__':
    bin = bin_analysis(0)
    summary_pca = pd.DataFrame(bin.summary_PCA(3))
    bin.ch_idx(summary_pca)
    # pick a value for k-means
    kmeans = bin.kmeans(summary_pca, 4)
    bin.data["kmeans"] = kmeans

    h5 = h5py.File("rst.hdf5", mode='w')
    data = h5["data"].value
    h5.close()
    for i in range(4):
        temp = bin.data[bin.data["kmeans"] == i]
        distances_summary, indice_summary = bin.KNN(temp[bin.spec], 2)
        distances_summary = -np.sort(-np.sort(distances_summary[:, 1]))
        eps = np.percentile(distances_summary, 0.1) # 0.04677
