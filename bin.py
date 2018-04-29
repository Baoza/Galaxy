from _bin import *
import re
from sklearn.cluster import KMeans
import KNN
from kmeans import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import h5py

data = pd.read_csv("summary.csv")

# data.replace(to_replace=-9999, value=None, inplace=True)
emission_lines = ['Flux_HeII_3203', 'Flux_NeV_3345', 'Flux_NeV_3425',
                  'Flux_OII_3726', 'Flux_OII_3728', 'Flux_NeIII_3868',
                  'Flux_NeIII_3967', 'Flux_H5_3889', 'Flux_He_3970',
                  'Flux_Hd_4101', 'Flux_Hg_4340', 'Flux_OIII_4363',
                  'Flux_HeII_4685', 'Flux_ArIV_4711', 'Flux_ArIV_4740',
                  'Flux_Hb_4861', 'Flux_OIII_4958', 'Flux_OIII_5006',
                  'Flux_NI_5197', 'Flux_NI_5200', 'Flux_HeI_5875',
                  'Flux_OI_6300', 'Flux_OI_6363', 'Flux_NII_6547',
                  'Flux_Ha_6562', 'Flux_NII_6583', 'Flux_SII_6716',
                  'Flux_SII_6730']

emission_data = data[emission_lines]


# select all the columns from the data set
selected_columns = [column for column in data.columns
                    if not re.match(r'.*[Ee]rr.*', column)]

data_train = data[selected_columns].copy()

quantiles_list = [i*0.1 for i in range(11)]
quantile_z = list(data_train["specz"].quantile(
    quantiles_list, interpolation='linear'))


# bin the data
bin_data = KNN.bin_data(data_train=data_train, quantile=quantiles_list)

# parameter: i
i = 0
temp = bin_data[i].copy()
zero_map = zero_est(temp)
neg_map = negative_est(temp)

# drop columns that has more than 25% lost data
drop_list = []
for item in zero_map:
    if zero_map[item] > 25:
        drop_list.append(item)

# cur_data stores the whole data set after deleting some of the columns
cur_data = bin_data[i].copy()
cur_data = cur_data.drop(drop_list, axis=1)
cur_data = cur_data.replace(-9999, 0)
cur_data = cur_data[np.all(cur_data != 0, axis=1)]

spec_data = cur_data.iloc[:, 52:len(cur_data.columns)]
summary_data = cur_data.iloc[:, 7:52]


# store data into h5py
# because it would ran out of RAM
h5 = h5py.File("bin"+str(i)+".hdf5", 'w')
h5.close()

h5 = h5py.File("bin"+str(i)+".hdf5", 'r+')
row_spec, col_spec = spec_data.shape
h5.create_dataset("spec_data", (row_spec, col_spec-1), np.float32)
h5.create_dataset("summary_data", summary_data.shape, np.float32)

# take ratio
# argmax = 'Flux_OII_3728'

mean_spec = np.mean(spec_data, axis=0)
max_spec_idx = np.argmax(mean_spec)
for k, name in enumerate(spec_data.columns):
    print("process:", name)
    if name == max_spec_idx:
        k -= 1
        continue
    temp = spec_data[name] / spec_data[max_spec_idx]
    h5["spec_data"][:, k] = temp

for k, name in enumerate(summary_data.columns):
    print("process:", name)
    temp = summary_data[name]
    h5["summary_data"][:, k] = temp
h5.close()

