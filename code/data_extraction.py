# Crop data based on activity time of each participant

import pandas as pd
import numpy as np
import datetime
import os
import re
import matplotlib.pyplot as plt

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

for file in activity_files:
    # read time 
    # participant data
    pData = pd.read_csv("./../data/mock study/Participant/" + file)
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
    patData.to_csv(save_dir + file.split(".")[0] + ".csv", index = False)



