import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def bin_data(data_train, quantile):
    bin_data = []
    for i in range(len(quantile)-1):
        bin_data.append(data_train.loc[
            (quantile[i] <= data_train["specz"]) &
            (data_train["specz"] < quantile[i+1])])
    return bin_data


#data_train_dered = data_train[["objid", "deVRad_u", "deVRad_g", "deVRad_r", "deVRad_i", "deVRad_z", "specz"]]
#bin_dered = bin_data(data_train_dered, quantile_z)


def knn_across_bins(data, bin1, bin2):
    """
    The data is a list of all the binned numpy array
    bin1: an int suggest the bin number with smaller specz
    bin2: an int suggest the bin number with bigger specz
    return:
      obj: a list of list given the Sosies' pair of their objid
    """
    data_bin1 = data[bin1]
    data_bin2 = data[bin2]
    closest = []

    for i in range(data_bin1.shape[0]):
        distance = 0
        for feature in ["deVRad_u",  "deVRad_g",  "deVRad_r",  "deVRad_i",  "deVRad_z"]:
            distance += abs(data_bin2[feature]
                            - data_bin1[feature].iloc[i])
        closest.append(data_bin2["objid"].loc[distance.idxmin()])
    obj = [(row[0], row[1]) for
           row in zip(*[list(data_bin1["objid"]), closest])]
    return obj


"""
#lst = knn_across_bins(bin_dered, 1, 4)
#lst2 = knn_across_bins(bin_dered, 1, 7)

x = np.linspace(1, 4, 100)
y = x**(3/2)
plt.plot(x, y)
plt.xlabel("Distance Ratio(apparent luminosity)")
plt.ylabel("Velocity(redshift)")
plt.title("Objective")
plt.axvline(x=1.2, ymax=1.2**(3/2)/4**(3/2), color="b")
plt.axvline(x=2, ymax=2**(3/2)/4**(3/2), color="b")
plt.axvline(x=3, ymax=3**(3/2)/4**(3/2), color="b")
plt.axvline(x=3.5, ymax=3.5**(3/2), color="b")
plt.show()

"""
