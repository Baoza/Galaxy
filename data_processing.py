import numpy as np
import pandas as pd
import re
import KNN
import h5py

##
# Define Constant Value Here ##
##

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

# type of magnitude used in later analysis
# Mag_type = ['cModelMag', 'fiberMag', 'deVMag', 'expMag', 'modelMag', 'dered']

Mag_type = ['cModelMag']

##
# Assistant Function
##


def zero_est(df):
    """
    check the invalid estimates of emission lines
    output a percentage of missing estimates
    """
    invalid_map = {}
    print("\n")
    for column in df.columns:
        print("{0}:{1}%".format(
            column, round(100*np.sum(df[column] == 0)/df.shape[0], 2)))
        invalid_map.update({column: round(100*np.sum(df[column] == 0)/df.shape[0], 2)})
    return invalid_map


def data_processing(emission_cols, num_bin=10, tol=25, csv_name="summary.csv"):
    """
    :param emission_cols: list of col_name of fluxes
    :param num_bin:
    :param tol: tolerant at most tol% missing data in one variable
    :param csv_name: the name of csv that needs processing

    usage: convert part of the data into ratio, store the results
    """
    data = pd.read_csv(csv_name)

    cols = [column for column in data.columns \
            if not re.match(r'.*[Ee]rr.*', column)]

    summary_cols = ['specObjID', 'specz', 'cModelMag_u', 'cModelMag_g', 'cModelMag_r',
                    'cModelMag_i', 'cModelMag_z', 'expAB_u', 'expAB_g', 'expAB_r', 'expAB_i', 'expAB_z']

    # STEP 1: divide all the data into 10 bins
    quantiles_list = [i * 0.1 for i in range(num_bin+1)]
    bin_data = KNN.bin_data(data_train=data, quantile=quantiles_list)

    # STEP 2:
    # for each bin, convert some columns into ratio and
    # store all the values in h5py
    for i in range(num_bin):
        print("process bin:", i)
        temp = bin_data[i]

        new, tol_cols = _data_processing(temp, summary_cols, tol, emission_cols)

        h5 = h5py.File("bin" + str(i) + ".hdf5", 'w')
        h5.create_dataset("data", data=new)
        h5.create_dataset("columns", data=repr(tol_cols))
        h5.close()


def _data_processing(temp, summary_cols, tol=0.25, emission_cols=emission_cols):
    zero_map = zero_est(temp)

    # STEP 2.1: find cols that have more than tol% missing data
    drop_list = []
    for item in zero_map:
        if zero_map[item] > tol:
            drop_list.append(item)

    # cols regarding fluxes to be included in later analysis
    spec_cols = [col for col in emission_cols if col not in drop_list]

    # drop whole lines of -9999
    temp = temp.drop(drop_list, axis=1)
    temp = temp[np.all(temp > 0, axis=1)]

    spec_data = temp[spec_cols]
    summary_data = temp[summary_cols]

    mean_spec = np.mean(spec_data, axis=0)
    max_spec_idx = np.argmax(mean_spec)

    temp = temp[temp[max_spec_idx] != 0]

    new = temp.copy()
    idx = 0
    for k, name in enumerate(spec_data.columns):
        print("process:", name)
        if name == max_spec_idx:
            continue
        else:
            new[name] = spec_data[name] / spec_data[max_spec_idx]
            a = new[name].quantile(0.01)
            b = new[name].quantile(0.95)
            new[name] = new[name][new[name].between(a, b)]
            idx += 1


    tol_drop_list = []
    for mag in Mag_type:
        mag_g = summary_data[mag + '_g']
        tol_drop_list.append(mag + '_g')
        for band in ['_u', '_r', '_i', '_z']:
            new[mag + band] = summary_data[mag + band] / mag_g
        new = new.drop(mag + '_g', axis=1)

    tol_cols = summary_cols + spec_cols
    tol_cols = [col for col in tol_cols if col != max_spec_idx and col not in tol_drop_list]

    new = new[tol_cols]
    return new, tol_cols


def read_h5py(bin_id=0):
    """
    :param bin:
    :param pd_true: true if we need pandas.DataFrame
                    false if we only need numpy
    :return: read data from h5py files and return the value
    """
    h5 = h5py.File("bin"+str(bin_id)+".hdf5", mode='r')
    data = h5["data"].value
    col_name = h5["columns"].value
    col_name = col_name[3:len(col_name) - 1]
    col_list = col_name.split("', '")
    col_list[len(col_list)-1] = col_list[len(col_list)-1][0:-1]
    h5.close()

    df = pd.DataFrame(data)
    df.columns = col_list
    df = df.dropna(0)
    return df





