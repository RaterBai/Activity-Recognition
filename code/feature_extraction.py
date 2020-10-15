# feature extraction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

INPUT_DIR = "./../result/"
FREQ = 30
files = []
for file in os.listdir(INPUT_DIR):
    if 'final' in file:
        files.append(file)

class FeatureExtraction:
    def __init__(self, input, freq):
        # initialize with a 2D numpy array only contains X, Y and Z axes.
        self.input = input
        self.freq = freq

    def calVm(self):
        self.vm = np.sqrt(np.sum(np.square(self.input), axis=1))
        
    def calAngle(self):
        # make sure the first axis is X-axis
        self.angle = 90*np.arcsin(self.input[:, 0]/self.vm)/(np.pi/2)
    
    def calFFT(self):
        # perform FFT and save result
        # scaled magnitude, scaled by sqrt(length of VM)
        # Eliminate upper half of the frequencies and strength, don't know why
        self.w = np.fft.fft(self.vm)[0:int(np.ceil(len(self.vm)/2))]/np.sqrt(len(self.vm))
        self.freqs = np.fft.fftfreq(len(self.vm))[0:int(np.ceil(len(self.vm)/2))] * self.freq
        
        # Remove first element to eliminate DC
        self.w = self.w[1:]
        self.freqs = self.freqs[1:]
        
    def getClosestIndexLeft(self, arr, val):
        i = 0
        while(arr[i] < val):
            i += 1
            
        if(arr[i+1] == val):
            return(i+1)
        
        if(i == 0):
            i += 1
        
        return(i-1)
        
    def getClosestIndexRight(self, arr, val):
        i = 0
        while(arr[i] < val):
            i += 1
        
        if(arr[i+1] == val):
            return(i+1)
        
        if(i == 0):
            i += 1
        
        if(i == len(arr)):
            i -= 1;
        
        return(i)
    
    def mvm(self):
        # mean of vector magnitude
        return(np.mean(self.vm))
    
    def sdvm(self):
        # standard deviation of vector magnitude
        return(np.std(self.vm))
    
    def mangle(self):
        # mean of angles
        return(np.mean(self.angle))
    
    def sdangle(self):
        # standard deviation of angles
        return(np.std(self.angle))
    
    def p625(self):
        # Percentage of the power of the vector 
        # magnitude that is in 0.6-2.5Hz 
        
        # first need to find first index > 0.6 and last index < 2.5
        point6Hz = self.getClosestIndexLeft(self.freqs, 0.6)
        twopint5Hz = self.getClosestIndexRight(self.freqs, 2.5)
        numerator = np.sum(abs(w[point6Hz+1: twopint5Hz+1]))
        denominator = np.sum(abs(w))
        return(numerator/denominator)
        
    def dominantFrequency(self):
        i = np.argmax(abs(self.w))
        dom_freq = self.freqs[i]
        dom_freq_hz = abs(dom_freq)
        return dom_freq_hz
    
    def fpdf(self):
        # Fraction of power in vector magnitude at
        # dominant frequency
        i = np.argmax(abs(self.w))
        fraction = abs(self.w[i])/np.sum(abs(np.delete(self.w, i)))
        return fraction
    
    def plot(self):
        self.calVm()
        self.calFFT()
        plt.plot(self.freqs, abs(self.w))
        plt.show()
        
    def getFeatures(self):
        
        # calculate Vector Magnitude
        self.calVm()
        mvm = self.mvm()
        sdvm = self.sdvm()
        
        # FFT
        self.calFFT()  # perform the calculations first
        df = self.dominantFrequency()
        p625 = self.p625()
        fpdf = self.fpdf()
        
        # calculate Angle
        self.calAngle()
        mangle = self.mangle()
        sdangle = self.sdangle()
        
        return([mvm, sdvm, df, p625, fpdf, mangle, sdangle])


# Read Participant files and create features
colnames = ["PID", "ActivityNumber", "group", "mvm", "sdvm", "df", "p625", "fpdf", "mangle", "sdangle"]
complete_data = []

for file in files:
    data = pd.read_csv(INPUT_DIR + file)
    participant_id = file.split("_")[0]
    activity_numbers = np.unique(data.ActivityNumber)
    Index = np.unique(data.Index)
    print(file)
    for i in activity_numbers:
        for j in Index:
            activity = data[(data.ActivityNumber == i) & (data.Index == j)]
            if(activity.shape[0] != 0):
                feature_extractor = FeatureExtraction(np.array(activity.loc[:, ["accelX", "accelY", "accelZ"]]), FREQ)
                features = feature_extractor.getFeatures()
                if i in [1, 2, 3, 4, 5, 6]:
                    group = 0  # non face touching
                else:
                    group = 1  # face touching
    
                features = [int(participant_id), i, group] + features
                complete_data.append(features)

data_created = pd.DataFrame(complete_data, columns = colnames)

# Physical Activity data
pa_data = data_created.loc[:, ["group", "mvm", "sdvm", "df", "p625", "fpdf", "mangle", "sdangle"]]

X = pa_data.loc[:, ["mvm", "sdvm", "df", "p625", "fpdf", "mangle", "sdangle"]]
Y = pa_data.loc[:, "group"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 123, shuffle=True)

# RF + GridSearch
params = {
    'n_estimators' : [100, 200, 300, 400],
    'max_depth' : [10, 20, 30, 40, 50],
    'min_samples_split' : [1, 2, 3, 4],
    'min_samples_leaf' : [1, 2, 3, 4, 5]
}
rf = RandomForestClassifier()
RF_gs = GridSearchCV(estimator=rf, param_grid=params, scoring='roc_auc', cv = 5,verbose=10)
RF_gs.fit(X_train, y_train)
estimator = RF_gs.best_estimator_
sum(estimator.predict(X_train) == y_train)
sum(estimator.predict(X_test) == y_test)

