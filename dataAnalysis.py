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
            if getDistance(fileName) == distance:
                allFileNames.append(fileName)
                distance = increaseDistance(distance)
                print(distance)
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


#**********************************************************************************
avgRSSIData = []
scanFolderName = 'No_Obstructions'
arrayAllFiles = getAllFolderFiles(scanFolderName)
#for file in arrayAllFiles:
#    avgRSSIData.append(avgRSSI(getRSSIData(scanFolderName, file)))
#print(avgRSSIData)
print("lol")
#print(arrayAllFiles)
