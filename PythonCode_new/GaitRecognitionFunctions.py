# -*- coding: utf-8 -*-
"""
Raw data (mat files): there are 17 labeled activities --> 0=none, 1=walking, 2=walking with transition, 3=shuffling,
    4=stairs(ascending), 5=stairs(descending), 6=standing, 7=sitting, 8=lying, 9=transition,10=leaning,11=undefined,
    12=jumping, 13=dynamic, 14=static, 15=shake, 16=picking, 17=kneeling
Aim_label: 1, walking
0-10 columns in each subject's signal: 0-2 ACCxyz;3-5 GYRxyz; 6-8 MAGxyz; 9 activity_Labels; 10, Time
Aim for this code: identify gait episodes and non-gait episodes with high performance
"""

import os
import re
import pickle
import openpyxl
import h5py  # use HDF reader for matlab v7.3 files
import numpy as np
import pandas as pd
import scipy.io as spio
import data_augmentation as DA
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix



recordInitializeWeights = True
useInitializeWeights = True


# %% 1. load the data and binary y_data into walking and non-walking
def delete_useless_label(DataX, DataY, DataY_binary, groups, del_labels):
    """
    delete activities "0=none" and "11=undefined" to save memory
    """
    loc_total = []
    for label in del_labels:
        loc = np.concatenate(np.array(np.where(DataY == label)))
        loc_total.append(loc)
    row_del = np.sort(np.hstack(loc_total))
    DataX_new = np.delete(DataX, row_del, 0)
    DataY_binary_new = np.delete(DataY_binary, row_del, 0)
    groups_new = np.delete(groups, row_del, 0)
    return DataX_new, DataY_binary_new, groups_new


def dichotomy_labels(y_data, aim_label):
    """
    :param y_data: activity labels for each subjects
            aim_label: the label we want is label '1' , representing walking
    :return: y_dataDichotoom: labels with only walking '1' and non_walking '0'
    """
    y_dataDichotoom = [0] * len(y_data)
    for indx in range(len(y_data)):
        if y_data[indx] == aim_label:
            y_dataDichotoom[indx] = aim_label
    return y_dataDichotoom

def read_matfiles(dataDir,VarName,input_axis):
    """
    Read the matfiles in a folder, which do not have y-labels.
    For unsupervised data
    :param dataDir:
    :param VarName:
    :param input_axis:
    :return:
    """
    filenames = []
    DataX = []
    data_all = []
    subject_number = []
    for file in os.listdir(dataDir):
        filenames.append(file)
        matpath = os.path.join(dataDir, file)
        if "Store" not in matpath:
            item = spio.loadmat(matpath)
            data_all.append(item[VarName])
            if input_axis == 3:
                DataX.append(item[VarName][:, 0:3])  # acc
            elif input_axis == 6:
                DataX.append(item[VarName][:, 0:6])  # acc&gyroscope
            else:
                print("Invalid Input Axis")

    p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
    if re.search(p, file) is not None:
        for catch in re.finditer(p, file):
            subject_number.append(int(catch[0]))
            # print('P' + catch[0])  # catch is a match object
            break


def load_matfiles(dataDir, aim_label, input_axis, del_labels):
    """
    :param: Name of the folder where the data is located
            each subject has a mat file, each mat file has a variable 'signal';
    :function: load a list of .mat data from a folder 'act17_mat'
               add column11: subject order number by using np.c_
    :note: the filenames is in a random order, not the subject number,so we extract no.sub from filenames
    """
    data_all = []
    filenames = []
    DataX = []
    DataY = []
    DataY_binary = []
    groups = []
    subject_number = []

    for file in os.listdir(dataDir):
        filenames.append(file)
        matpath = os.path.join(dataDir, file)
        if "Store" not in matpath:
            item = spio.loadmat(matpath)
            data_all.append(item["signal"])
            if input_axis == 3:
                DataX.append(item["signal"][:, 0:3])  # acc
            elif input_axis == 6:
                DataX.append(item["signal"][:, 0:6])  # acc&gyroscope
            else:
                print("Invalid Input Axis")
            DataY.append(item["signal"][:, 9])  # activity labels

        p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'
        if re.search(p, file) is not None:
            for catch in re.finditer(p, file):
                subject_number.append(int(catch[0]))
                # print('P' + catch[0])  # catch is a match object
                break

    for indx in range(len(data_all)):
        DataY_binary.append(dichotomy_labels(DataY[indx], aim_label))
        no = subject_number[indx]
        group_col = [no] * data_all[indx].shape[0]
        groups.append(group_col)
        data_all[indx] = np.c_[data_all[indx], group_col]

    DataX = np.array(DataX, dtype=object)
    DataY = np.array(DataY, dtype=object)
    DataY_binary = np.array(DataY_binary, dtype=object)
    groups = np.array(groups, dtype=object)
    print(filenames)

    DataX_tmp = np.concatenate((DataX[:]))
    DataY_tmp = np.concatenate(DataY[:])
    DataY_binary_tmp = np.concatenate(DataY_binary[:])
    groups_tmp = np.concatenate(groups[:])

    if del_labels:
        DataX, DataY_binary, groups = delete_useless_label(DataX_tmp, DataY_tmp, DataY_binary_tmp, groups_tmp,
                                                           del_labels)
    else:
        DataX, DataY_binary, groups = DataX_tmp, DataY_binary_tmp, groups_tmp

    return DataX, DataY_binary, groups, filenames, subject_number


# %% 2 Data preparation
# 2.1 segment data
def windows(data_size, size, percentage):
    start = 0
    while start < data_size:
        yield int(start), int(start + size)
        start += int(size * percentage)


def segment_signal(X, y, groups, window_size, percentage):
    """ For supervised model training, or external validation with y labels
        segment signal X, activity label y, and groups (subjects numbers) into windows
        For signal X, remove orientation per window
    :param window_size: seconds*sample rate
    :return: X: windows of signals (3acc,or 3acc&3gyr)
             y: windows of labels
             Groups: windows of subjects number
             the function only remains the windows with one act label (DataY)
    """
    X_win, y_win, groups_win, y_final = list(), list(), list(), list()
    for (start, end) in windows(len(X), window_size, percentage):
        if len(X[start:end]) == window_size and len(np.unique(y[start:end])) == 1 and len(
                np.unique(groups[start:end])) == 1:
            X_win.append((X[start:end]) - np.mean(X[start:end], axis=0))
            y_win.append(np.unique(y[start:end]))
            groups_win.append(np.unique(groups[start:end]))
    X_final = np.array(X_win)
    y_win = np.array(y_win)
    y_final = to_categorical(y_win)
    groups_win = np.array(groups_win)
    groups_final = np.concatenate(groups_win)
    return X_final, y_final, groups_final


# 2.2 balance data on windows
def down_sampling_win(X, y, groups, i, output_path, data_type):
    """
    after splitting data into windows,
    remain windows of non-walking and walking in specific ratio (here we remain 1:1) in each subject group
    :param groups: the number of subjects
    :param X: the data of 3acc in windows
    :param y: the label of walking 1 and non-walking in windows
    :param groups: the number of subjects in windows
    :return: the balanced X, y(2 labels) and groups
    for subject=1, length of walking sampling points in DataX is 91453 (457.265s),
    non-walking is 795236
    """
    DataX_blc = []
    DataY_blc = []
    DataGroups_blc = []
    Nwin_walking = 0
    Nwin_nonwalking = 0
    Nwin_blc_walking = 0
    Nwin_blc_nonwalking = 0

    for indx in np.unique(groups):
        ind = groups == indx
        X_isub = X[ind, :, :]
        y_isub = y[ind, :]
        groups_isub = groups[ind]

        walking_indices = np.where(y_isub[:, 1] == 1)[0]
        nonwalking_indices = np.where(y_isub[:, 1] == 0)[0]
        nonwalking_indices_blc = shuffle(nonwalking_indices, random_state=42)[:len(walking_indices)]

        all_included = np.in1d(nonwalking_indices_blc, nonwalking_indices)
        if np.all(all_included):
            print("All elements of nonwalking_indices_blc are included in nonwalking_indices.")
        else:
            print("Not all elements of nonwalking_indices_blc are included in nonwalking_indices.")

        selectedIndices = np.hstack((walking_indices, nonwalking_indices_blc))
        selectedIndices = np.sort(selectedIndices)

        X_blc = X_isub[selectedIndices, :, :]
        y_blc = y_isub[selectedIndices, :]
        groups_blc = groups_isub[selectedIndices]

        DataX_blc.append(X_blc)
        DataY_blc.append(y_blc)
        DataGroups_blc.append(groups_blc)

        Nwin_walking = Nwin_walking + walking_indices.__len__()
        Nwin_nonwalking = Nwin_nonwalking + nonwalking_indices.__len__()
        Nwin_blc_walking = Nwin_blc_walking + np.where(y_blc[:, 1] == 1)[0].__len__()
        Nwin_blc_nonwalking = Nwin_blc_nonwalking + np.where(y_blc[:, 1] == 0)[0].__len__()

        pathToNumWins = os.path.join(output_path, 'number of windows')
        if not os.path.exists(pathToNumWins):
            os.makedirs(pathToNumWins)
        filepath = os.path.join(pathToNumWins, f'NumWins_{data_type}_{i}.txt')
        with open(filepath, 'a') as file:
            print(f'Repeat{i}', file=file)
            print(f'Subject {indx} has walking windows: {walking_indices.__len__()}', file=file)
            print(f'Subject {indx} has non-walking windows:{nonwalking_indices.__len__()}', file=file)
            print(f'Subject {indx} has selected windows: {len(selectedIndices)}', file=file)
            print(f'Subject {indx} has balanced walking windows: {np.sum(y_blc[:, 1] == 1)}', file=file)
            print(f'Subject {indx} has balanced non-walking windows: {np.sum(y_blc[:, 1] == 0)}', file=file)

    DataX_blc = np.array(DataX_blc, dtype=object)
    DataX_blc = np.concatenate(DataX_blc[:], dtype=object)
    DataY_blc = np.array(DataY_blc, dtype=object)
    DataY_blc = np.concatenate(DataY_blc[:])
    DataGroups_blc = np.array(DataGroups_blc)
    DataGroups_blc = np.concatenate(DataGroups_blc[:])

    print(f"length of walking is {Nwin_walking}")
    print(f"length of non-walking is {Nwin_nonwalking}")
    print(f"length of balanced walking is {Nwin_blc_walking}")
    print(f"length of balanced non-walking is {Nwin_blc_nonwalking}")
    return DataX_blc, DataY_blc, DataGroups_blc


# 2.3 split data into train and test dataset, split train dataset as train and validation dataset
def split_GroupNfolds(X_data, y_data, Groups, Nfolds):
    """
    :param X_data: 3acc,3gyr (no windows)
    :param y_data: labels of activities
    :param Nfolds: the number of folds we want to split, when Nfolds=20, it's leave-one-out cross validation(LOOCV)
                   Nfolds = 4  # train:test = 0.76:0.24
    :param Groups: subjects' numbers
    :return:
    """
    kf = GroupKFold(n_splits=Nfolds)
    index_t_t = list(enumerate(kf.split(X_data, groups=Groups)))
    n_random = np.random.randint(len(index_t_t))
    train_index = index_t_t[n_random][1][0]
    test_index = index_t_t[n_random][1][1]
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    groups_train, groups_test = Groups[train_index], Groups[test_index]
    print(f"Train_group = {np.unique(groups_train)}")
    print(f" Test_group ={np.unique(groups_test)}")
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def split_LeaveOneOut(X_t_v, y_t_v, groups_t_v):
    """
    using leave-one-group-out method to split training and validating dateset
    :param X_t_v: X_data,containing training and validating dataset
    :param y_t_v: y_data, training and validating dataset
    :param groups_t_v: subjects part
    :return: training dataset and validating dataset
    """
    logo = LeaveOneGroupOut()
    index_t_v = list(enumerate(logo.split(X_t_v, y_t_v, groups_t_v)))
    n_random = np.random.randint(len(index_t_v))
    train_index = index_t_v[n_random][1][0]
    val_index = index_t_v[n_random][1][1]

    X_train, X_val = X_t_v[train_index], X_t_v[val_index]
    y_train, y_val = y_t_v[train_index], y_t_v[val_index]
    groups_train, groups_val = groups_t_v[train_index], groups_t_v[val_index]
    return X_train, X_val, y_train, y_val, groups_train, groups_val


# %% 3. fit and evaluate a CNN model
# source:https://github.com/apachecn/ml-mastery-zh/blob/master/docs/dl-ts/cnn-models-for-human-activity-recognition-time-series-classification.md
def plot_loss(history, i, output_path):
    plt.figure()
    plt.title('Loss/BinaryCrossentropy')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    loss_plot_path = os.path.join(output_path, 'loss plot')
    if not os.path.exists(loss_plot_path):
        os.makedirs(loss_plot_path)
    plt.savefig(f'{output_path}/loss plot/loss{i}.png')


def cnn_model(x_t, y_t, i, output_path):
    """define the Convolutional neural network(CNN) Architecture"""
    n_timesteps, n_features, n_outputs = x_t.shape[1], x_t.shape[2], y_t.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    # Define EarlyStopping callback to monitor training loss
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    modelweights = []
    pathToInitWeights = os.path.join(output_path, 'InitialWeights')
    if not os.path.exists(pathToInitWeights):
        os.makedirs(pathToInitWeights)

    for indx in range(0, len(model.layers)):
        modelweights.append(model.layers[indx].get_weights())

    if recordInitializeWeights:
        f = open(pathToInitWeights + '/InitWeights' + str(i) + '_.pckl', 'wb')
        pickle.dump(modelweights, f)
        f.close()

    if useInitializeWeights:
        f = open(pathToInitWeights + '/InitWeights' + str(i) + '_.pckl', 'rb')
        modelWeightstesting = pickle.load(f)
        f.close()
        for indx in range(0, len(model.layers)):
            a = model.layers[indx].get_weights()
            if not a:
                print('layer is empty')
            else:
                model.layers[indx].set_weights([modelWeightstesting[indx][0], modelWeightstesting[indx][1]])
                print('weights initialized for layer: ' + str(indx))
    return model, early_stopping


def fit_and_evaluate(x_t, x_val, y_t, y_val, x_test, y_test, i, output_path):
    """
    :param x_t: X_training
    :param x_val: X_validation
    :param y_t: y_training
    :param y_val: y_validation
    :param x_test: X_test
    :param y_test: y_test
    :return: validation_results, test_results
        Here's what the typical end-to-end workflow looks like, consisting of:
        Training; Validation on a holdout set generated from the original training data; Evaluation on the test data

        train the model by slicing the data into "batches" of size batch_size,
        and repeatedly iterating over the entire dataset for a given number of epochs.
    """
    print("Fit model on training data")
    x_t = x_t.astype(np.float32)
    y_t = y_t.astype(np.float32)
    x_val = x_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    verbose, epochs, batch_size = 1, 30, 32
    model, early_stopping = cnn_model(x_t, y_t, i, output_path)
    history = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(x_val, y_val),
                        callbacks=[early_stopping])
    plot_loss(history, i, output_path)

    print("validate loss, Validate acc:", model.evaluate(x_val, y_val, batch_size=128))
    print("test loss, test acc: ", model.evaluate(x_test, y_test, batch_size=128))

    y_val_predict = model.predict(x_val).round()
    score_val = model_performance(y_val, y_val_predict)
    print("CNN_model results on validation data: ", score_val)

    y_test_predict = model.predict(x_test).round()
    score_test = model_performance(y_test,
                                   y_test_predict)  # try to use from sklearn.metrics import classification_report
    print("CNN_model results on test data: ", score_test)
    return model, history, score_val, score_test


def model_performance(y_true, y_pred):
    """ WE should use only 1 column containing all labels as input"""
    acc = accuracy_score(y_true[:, 1], y_pred[:, 1])
    preci = precision_score(y_true[:, 1], y_pred[:, 1])
    rec = recall_score(y_true[:, 1], y_pred[:, 1])
    f1 = f1_score(y_true[:, 1], y_pred[:, 1])
    tn, fp, fn, tp = confusion_matrix(y_true[:, 1], y_pred[:, 1].round()).ravel()
    spec = tn / (fp + tn)
    score = pd.DataFrame([[acc, preci, rec, f1, spec, tn, fp, fn, tp]],
                         columns=['Accuracy', 'Precision', 'Sensitivity', 'F1', 'Specificity', 'tn', 'fp', 'fn', 'tp'])
    return score


def run_model(X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test, window_size,
              overlap_rate, i, output_path):
    model_history = []
    # segment Xy data into windows and categorical y_data
    print("## 2) segment data into windows")
    X_train_win, y_train_win, groups_train_win = segment_signal(X_train, y_train, groups_train, window_size,
                                                                overlap_rate)
    X_val_win, y_val_win, groups_val_win = segment_signal(X_val, y_val, groups_val, window_size, overlap_rate)
    X_test_win, y_test_win, groups_test_win = segment_signal(X_test, y_test, groups_test, window_size, overlap_rate)

    # down_sampling walking and non_walking data into 1:1
    print("## 3) down_sampling walking and non_walking of train and validate data into 1:1")
    print("## 3.1) For training data:")
    X_train_blc, y_train_blc, groups_train_blc = down_sampling_win(X_train_win, y_train_win, groups_train_win, i,
                                                                   output_path, 'train')
    print("## 3.2) For validating data:")
    X_val_blc, y_val_blc, groups_val_blc = down_sampling_win(X_val_win, y_val_win, groups_val_win, i, output_path,
                                                             'vali')

    print("## 4) running the model")
    model, history, score_val, score_test = fit_and_evaluate(X_train_blc, X_val_blc, y_train_blc, y_val_blc, X_test_win,
                                                             y_test_win, i, output_path)
    model_history.append(history)
    final_epoch = len(history.history['loss'])
    print("===========" * 12, end="\n\n\n")
    return model, model_history, final_epoch, score_val, score_test


def train(repeats, DataX, DataY_binary, groups, Nfolds, augmentation_methods, input_axis, Nfolds_val, fs, window_size,
          overlap_rate, ModelsInfoDir, ModelsSaveDir, ModelResultDir):
    """

    :param repeats:
    :param DataX:
    :param DataY_binary:
    :param groups:
    :param Nfolds:
    :param augmentation_methods:
    :param input_axis:
    :param Nfolds_val:
    :param fs:
    :param window_size:
    :param overlap_rate:
    :param ModelsInfoDir:
    :param ModelsSaveDir:
    :param ModelResultDir:
    :return:
    """
    # %%  train the model
    # %%    Steps: split train_val&test, split train&val, +/-DA, run the model(segment, down-sample, fit model,evaluate)
    Scores_val = list()
    Scores_test = list()

    for i in range(1, repeats+1):
        print("===========" * 6, end="\n\n\n")
        print('Repeat ', i)
        X_train_val, X_test, y_train_val, y_test, groups_train_val, groups_test = split_GroupNfolds(DataX, DataY_binary,
                                                                                                    groups, Nfolds)
        print(f'## 2) DA = {augmentation_methods}')
        print("## 3) split train and validate data")
        if augmentation_methods is not None:
            X_train_val_all, y_train_val_all, groups_train_val_all = use_data_augmentation(X_train_val, y_train_val,
                                                                                           groups_train_val, fs,
                                                                                           augmentation_methods,
                                                                                           input_axis)
            X_train, X_val, y_train, y_val, groups_train, groups_val = split_GroupNfolds(X_train_val_all,
                                                                                            y_train_val_all,
                                                                                            groups_train_val_all,
                                                                                            Nfolds_val)
        elif augmentation_methods is None:
            X_train, X_val, y_train, y_val, groups_train, groups_val = split_GroupNfolds(X_train_val, y_train_val,
                                                                                         groups_train_val, Nfolds_val)
        print("## 4) training the model")
        CNN_model, model_history, final_epoch, score_val_i, score_test_i = run_model(X_train, X_val, X_test,
                                                                                     y_train, y_val, y_test,
                                                                                     groups_train, groups_val,
                                                                                     groups_test,
                                                                                     window_size, overlap_rate,
                                                                                     i, ModelsInfoDir)
        # combine results
        score_val_i['No.'] = i
        score_test_i['No.'] = i
        score_val_i['ModelEpoch'] = final_epoch
        score_test_i['ModelEpoch'] = final_epoch
        Scores_val.append(score_val_i)
        Scores_test.append(score_test_i)

        # save
        CNN_model.save(f'{ModelsSaveDir}/CNNmodel_{i}.h5')
        score_val_i.to_csv(f'{ModelResultDir}/scores_val.txt', mode='a', sep='\t', index=False, header=not i)
        score_test_i.to_csv(f'{ModelResultDir}/scores_test.txt', mode='a', sep='\t', index=False, header=not i)

        filepath = os.path.join(ModelResultDir, 'file_groups.txt')
        with open(filepath, 'a') as file:
            print(f'Repeat{i}', file=file)
            print(f"Train_group = {np.unique(groups_train)}", file=file)
            print(f"Test_group ={np.unique(groups_test)}", file=file)
            print(f"Validate_group ={np.unique(groups_val)}", file=file)

    save_dataframe_xlsx(Scores_val, ModelResultDir, 'scores_val.xlsx')
    save_dataframe_xlsx(Scores_test, ModelResultDir, 'scores_test.xlsx')


def save_dataframe_xlsx(data, output_path, filename):
    data = pd.concat(data, ignore_index=True)  # remain only one header
    output_xlsx_path = os.path.join(output_path, filename)
    if os.path.isdir(output_xlsx_path):
        output_xlsx_path = os.path.join(output_xlsx_path)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    workbook.save(output_xlsx_path)
    data.to_excel(output_xlsx_path, index=False)


def use_data_augmentation(x, y, groups, fs, augmentation_methods=None, input_axis=None):
    """ Determine whether use the Data Augmentation methods and which methods
        The magnitude number of DA can be adjusted
    :return: The final data before putting model [raw data +/- DA]

    """
    augmented_data = []

    # Apply selected augmentation methods
    if 'noise' in augmentation_methods:
        x_noise, y_noise, groups_noise = DA.data_with_noise(x, y, groups, [0.01, 0.02])
        augmented_data.extend([x_noise, y_noise, groups_noise])

    if 'scaling' in augmentation_methods:
        x_scaling, y_scaling, groups_scaling = DA.data_with_scaling(x, y, groups, [0.3, 0.4])
        augmented_data.extend([x_scaling, y_scaling, groups_scaling])

    if 'interpolation' in augmentation_methods:
        x_intrp, y_intrp, groups_intrp = DA.data_with_intrp(x, y, groups, fs, [0.9, 1.1])
        augmented_data.extend([x_intrp, y_intrp, groups_intrp])

    if 'rotation' in augmentation_methods:
        if input_axis == 3:
            x_rot, y_rot, groups_rot = DA.rotation_acc(x, y, groups, [90])
        elif input_axis == 6:
            x_rot, y_rot, groups_rot = DA.rotation(x, y, groups, [90])
        else:
            print("Undefined Input Axis")
        augmented_data.extend([x_rot, y_rot, groups_rot])

    # Combine all augmented data
    X_all = np.concatenate([x] + augmented_data[::3], axis=0)
    y_all = np.concatenate([y] + augmented_data[1::3], axis=0)
    groups_all = np.concatenate([groups] + augmented_data[2::3], axis=0)
    return X_all, y_all, groups_all


## AIM 2: validate existing models
def load_txtdata(DataTxtDir, input_axis):
    """
    :param DataTxtDir: the folder of data with txt format
    :param input_axis: choose 3/6: only acceleration or combined with gyroscope
    :return:
    """
    # load external data (txt files)
    with open(f'{DataTxtDir}/filenames_balance', 'rb') as fp:
        filenames_sta = pickle.load(fp)
    with open(f'{DataTxtDir}/filenames_wk', 'rb') as fp:
        filenames_wk = pickle.load(fp)
    signals_sta = np.loadtxt(f'{DataTxtDir}/signals_sta.txt')
    signals_wk = np.loadtxt(f'{DataTxtDir}/signals_wk.txt')
    y_sta = np.loadtxt(f'{DataTxtDir}/y_sta.txt')
    y_wk = np.loadtxt(f'{DataTxtDir}/y_wk.txt')
    group_sta = np.loadtxt(f'{DataTxtDir}/group_sta.txt')
    group_wk = np.loadtxt(f'{DataTxtDir}/group_wk.txt')
    DataX = np.vstack((signals_wk,signals_sta))
    if input_axis == 3:
        DataX = DataX[:,0:3]
    elif input_axis == 6:
        DataX = DataX[:,0:6]

    DataY_tmp = np.vstack((y_wk.reshape(-1,1),y_sta.reshape(-1,1)))
    DataY = DataY_tmp.reshape(-1,)
    groups_tmp = np.vstack((group_wk.reshape(-1,1),group_sta.reshape(-1,1)))
    groups = groups_tmp.reshape(-1,)
    return DataX, DataY, groups


def validation(fs, window_size, overlap_rate, input_axis, model_path, DataX, DataY, groups, augmentation_methods, plotsingal, ExValScoresDir):
    #  data augmentation
    if augmentation_methods is not None:
        DataX_new, DataY_new, groups_new = use_data_augmentation(DataX, DataY, groups, fs,augmentation_methods,input_axis)
    elif augmentation_methods is None:
        DataX_new, DataY_new, groups_new = DataX, DataY, groups

    # segment data into windows
    X_win, y_win, groups_win = segment_signal(DataX_new, DataY_new, groups_new, window_size, overlap_rate)

    # load existing models from the folders and predict y labels in a loop
    scores_all = []
    for filename in os.listdir(model_path):
        if filename.endswith('.h5'):
            filename_without_externsion = os.path.splitext(filename)[0]
            parts = filename_without_externsion.split('_')
            model_number = parts[-1]

            file_path = os.path.join(model_path,filename)
            cnn_model = load_model(file_path)
            y_predict = cnn_model.predict(X_win).round()

            # remove the overlapping, although it does not influence the results
            overlap_samples = int(overlap_rate * window_size)
            X_rm_repeat = X_win[:, :overlap_samples, :]
            X = X_rm_repeat.reshape(-1, X_win.shape[-1])
            winsize_new = int(window_size-overlap_samples)
            y_predict_final = np.repeat(y_predict, winsize_new, axis=0)
            y_true_final = np.repeat(y_win, winsize_new, axis=0)
            groups_final = np.repeat(groups_win, winsize_new, axis=0)
            score = model_performance(y_true_final, y_predict_final)
            score = score.assign(model_number=model_number)
            scores_all.append(score)
            print('CNN model results of external dataset.\n', score)

    flattened_scores = [np.ravel(arr) for arr in scores_all]
    scores_export = pd.DataFrame(flattened_scores,
                          columns=['Accuracy', 'Precision', 'Sensitivity', 'F1', 'Specificity', 'tn', 'fp', 'fn', 'tp',
                                   'model_number'])
    print('CNN model results of external dataset.\n', scores_export)
    scores_export.to_excel(f"{ExValScoresDir}/scores_external_validation.xlsx", index=False)

    if plotsingal:
        plot_signal(X, y_predict_final, y_true_final, fs, ExValScoresDir,f'External validation dataset for model_{model_number}')

    return score, y_predict_final, y_true_final


def plot_signal(X, y_predict_final, y_true_final, fs, ExValScoresDir, which_datasets):
    time_seconds = np.arange(len(X)) / fs
    plt.figure(figsize=(19, 10))
    plt.plot(time_seconds, X[:, 0])
    plt.plot(time_seconds, y_predict_final[:, 1])
    plt.plot(time_seconds, -y_true_final[:, 1])  # true labels
    plt.title(f'Predicted and True Vertical Acceleration_{which_datasets}')  #accX
    plt.legend(['Signal', 'Predicted Label (wk=1)', 'True Label (wk=-1)'])
    plt.xlabel("Seconds", fontsize=20)
    plt.ylabel("Gravity Acceleration (9.8 m/s²)", fontsize=16)
    plt.tick_params(axis='x', labelsize=15, labelcolor='black', pad=10)
    plt.tick_params(axis='y', labelsize=15, labelcolor='black', pad=10)
    plt.savefig(f'{ExValScoresDir}/signal_AccxModel.png') # AccX is vertical acc


# AIM 3: unsupervised data
def load_matstrudata(data_path):
    """
    :param data_path: a mat file, containing a structure "dataset", where has multiple subjects' data
                      the fields of the stucture is 'name','baseline','post'
    :return:
    """
    data = h5py.File(data_path)
    list(data.keys())    #['#refs#', '#subsystem#', 'dataset']
    struArray = data['/dataset']
    data[struArray['name'][0,0]].value


def load_csvfile(csv_path):
    """
    :param csv_path: columns of the data: time, accX,accY,accZ
    :return:
    """
    dt = pd.read_csv(csv_path).values
    X = dt[:,-3:]
    return X


def predict_data_unsupervised(X, model_path, window_size, overlap_rate):
    """ how to achieve transfer learning 
    :param X: one subject's data (3-axis, or 6-axis) 
    :param model_path: the path of the responding cnn model .h5 (3-axis, or 6-axis)
    :param window_size: the same window size with the model 
    :param overlap_rate:  the overlapping rate in the windows
    :return: 
    """""
    X_win, groups_win, y_final = list(), list(), list()
    for (start, end) in windows(len(X), window_size, overlap_rate):
        if len(X[start:end]) == window_size:
            X_win.append((X[start:end]) - np.mean(X[start:end], axis=0))

    X_final = np.array(X_win)

    # load existing models
    cnn_model = load_model(model_path, compile=False)

    # predict y
    y_predict = cnn_model.predict(X_final)

    plt.plot(y_predict[:,1])
    plt.title('Possibility of walking')
    plt.xlabel('Sampling points')
    plt.ylabel('Possibility')
    plt.show()
    return X_final, y_predict
