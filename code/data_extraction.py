# Crop data based on activity time of each participant

import pandas as pd
import numpy as np
import datetime
import os
import re
import matplotlib.pyplot as plt

# frequence of the accelerometer data
FREQ = 30 
WINDOW_LENGTH = 10

accel_dir = "./../data/mock study/"
activity_dir = "./../data/mock study/Participant/"

accel_files = os.listdir(accel_dir)
accel_files_needed = []

save_dir = "./../result/"
# read participant information
activity_files = os.listdir(activity_dir)


for file in accel_files:
    if 'sensor' in file:
        accel_files_needed.append(file)
accel_files_needed = sorted(accel_files_needed)

# read all files and add them to one dataframe
data_list = []
for file in accel_files_needed:
    data = pd.read_json("./../data/mock study/"+file)
    data_list.append(data)
data = pd.concat(data_list)

# Given the input data, select certain period of data using start time and end time
def selectPeriod(start, end, input):
    start = pd.to_datetime(start, format="%d/%m/%Y %H:%M:%S%z")
    end = pd.to_datetime(end, format="%d/%m/%Y %H:%M:%S%z")
    return input[(input['timestamp'] > start) & (input['timestamp'] <= end)].copy()

def crop_by_window(data, window):
    length = window * FREQ
    data_list = []
    # if the last one miss more than 10%, remove them
    print("data.shape[0] : %d    length : %d" %(data.shape[0], length))
    if(data.shape[0]%length < length*9/10):
        for i in range(data.shape[0]//length):
            tmp = data[i*length: (i+1)*length].copy()
            tmp['Index'] = i
            tmp.reset_index()
            data_list.append(tmp)
    else:
        for i in range(int(np.ceil(data.shape[0]/length))):
            if(i != np.ceil(data.shape[0]/length)-1):
                tmp = data[i*length:(i+1)*length].copy()
            else:
                tmp = data[i*length:].copy()
            tmp['Index'] = i
            tmp.reset_index()
            data_list.append(tmp)
    retval = pd.concat(data_list)
    return(retval.copy())


for file in activity_files:
    #print(file)
    # read time 
    # participant data
    pData = pd.read_csv("./../data/mock study/Participant/" + file)
    #print(pData)
    # create patient data by cropping time
    data_list = []
    for i in range(pData.shape[0]):
        startTime = pData.iloc[i, 0]
        endTime = pData.iloc[i, 1]
        newData = selectPeriod(startTime, endTime, data)
        newData['ActivityNumber'] = pData.iloc[i, 2]
        data_list.append(newData)
    # selected participant data
    patData = pd.concat(data_list)
    print(patData.head())
    # apply window-length to crop data
    # For each activity 
    act_list = []  # activity list
    activities = np.unique(patData.ActivityNumber)

    for activity in activities:
        tmp = patData.query("ActivityNumber==%s"%str(activity))
        retval = crop_by_window(tmp, WINDOW_LENGTH)
        act_list.append(retval)

    finalData = pd.concat(act_list)

    finalData.to_csv(save_dir + file.split(".")[0] + "_final.csv", index = False)
    patData.to_csv(save_dir + file.split(".")[0] + ".csv", index = False)



