import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import seaborn as sns
import numpy as np
import random


def zero_est(df):

    """
    a brief function to check the invalid estimates of emission lines
    output a percentage of missing estimates
    """
    invalid_map = {}
    print("\n")
    for column in df.columns:
        print("{0}:{1}%".format(
            column, round(100*np.sum(df[column] == 0)/df.shape[0], 2)))
        invalid_map.update({column: round(100*np.sum(df[column] == 0)/df.shape[0], 2)})
    return invalid_map


def negative_est(df):
    missing_map = {}
    print("\n")
    for column in df.columns:
        print("{0}:{1}%".format(
            column, round(100*np.sum(df[column] < 0)/df.shape[0], 2)))
        missing_map.update({column: round(100*np.sum(df[column] < 0)/df.shape[0], 2)})
    return missing_map



def pie_chart(df):

    """
    This function returns a random galaxy pie chart plot from all the
    galaxy data frames.
    This takes cares of the case
    (etc that there are multiple NeIII emission lines)
    in the data frame and add them up to give a total estimate of the
    particular component.
    """

    idx = random.randint(0, df.shape[0])
    bool_index = df.loc[idx] != 0
    label = [column.split("_")[1] for column, ind in
             zip(df.columns, bool_index) if ind]
    values = df.loc[idx][bool_index]

    distinct_label = set(label)
    distinct_value = []
    for lab in distinct_label:
        current_value = 0
        for i, lab2 in enumerate(label):
            if lab2 == lab:
                current_value += values[i]
        distinct_value.append(current_value)

    plt.pie(distinct_value, labels=distinct_label,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.show()

