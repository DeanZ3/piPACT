import csv
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#returns array of first 100 RSSI values
#in file, real data starts on line 2, so 100th RSSI value is on line 101
def getRSSIData(folderName, file):
    fileName = os.path.join(folderName, file)
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        lineCount = 0
        data = [0] * 100
        for row in csv_reader:
            if lineCount == 0:
                lineCount += 1
            elif lineCount <= 100:
                data[lineCount - 1] = int(row[7])
                lineCount += 1
            else:
                break
        return data

def avgRSSI(allData):
    return sum(allData) / int(len(allData))

#returns distance in inches
def getDistance(fileName):
    distance = 0.0
    index = fileName.find('_')
    newName = fileName.replace('_', '', 1)
    distance = newName[index : newName.find('_')]
    return distance

def increaseDistance(distance):
    if distance == 59.5:
        return 62.0
    elif distance == 62.0:
        return 63.0
    elif distance == 143.5:
        return 144.0
    else:
        return distance + 3.5

def sortDistance(fileNames):
    distance = 0.0
    allFileNames = []
    while distance <= 144.0:
        for fileName in fileNames:
            if getDistance(fileName) == str(distance):
                allFileNames.append(fileName)
                distance = increaseDistance(distance)
    return allFileNames

#returns temperature in fahrenheit
def getTemp(fileName):
    temp = 0
    length = len(fileName)
    temp += int(fileName[length - 25]) * 10
    temp += int(fileName[length - 24])
    return temp

#returns "relative humidity" in percent form
def getHumidity(fileName):
    humidity = 0
    length = len(fileName)
    humidity += int(fileName[length - 22]) * 10
    humidity += int(fileName[length - 21])
    return humidity

#returns array of file names
def getAllFolderFiles(folderName):
    folderPath = os.path.join(r'C:\Users\jzhan\Desktop\piPACT', folderName)
    fileNames = os.listdir(folderPath)
    return sortDistance(fileNames)


"""
avgRSSIData = []
scanFolderName = 'No_Obstructions'
arrayAllFiles = getAllFolderFiles(scanFolderName)
for file in arrayAllFiles:
    avgRSSIData.append(avgRSSI(getRSSIData(scanFolderName, file)))
print(avgRSSIData)
"""

"""
df = pd.read_csv(r'No_Obstructions\scan_0.0_77_53_20200626T150621.csv')
df = df.replace('?', np.nan)
df = df.dropna()

df = df.drop(['MAJOR','MINOR','TX POWER'], axis=1)
X = df.drop('RSSI', axis=1)
y = df[['RSSI']]
print(X)
print("hi")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
"""

"""
df = pd.read_csv('mpg.csv')
df = df.replace('?', np.nan)
df = df.dropna()

df = df.drop(['name','origin','model_year'], axis=1)
X = df.drop('mpg', axis=1)
y = df[['mpg']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

reg = LinearRegression()
reg.fit(X_train[['horsepower']], y_train)

y_predicted = reg.predict(X_test[['horsepower']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))

reg = LinearRegression()
reg.fit(X_train[['horsepower','weight','cylinders']], y_train)
y_predicted = reg.predict(X_test[['horsepower','weight','cylinders']])
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('R²: %.2f' % r2_score(y_test, y_predicted))

fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('measured')
ax.set_ylabel('predicted')
plt.show()
"""
