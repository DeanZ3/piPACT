import csv
import os
import pandas

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

#can't import pandas
#def data(fileName):
#    file = pandas.read_csv(fileName)
#    for row in file:
#        if lineCount == 0:
#            lineCount += 1
#        elif lineCount <= 100:
#            data[lineCount - 1] = row[7]
#            lineCount += 1
#        else:
#            return data

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
    distance = 0.0
    allFileNames = []
    folderPath = os.path.join(r'C:\Users\jzhan\Desktop\piPACT', folderName)
    fileNames = os.listdir(folderPath)
    while distance <= 144.0:
        for fileName in fileNames:
            if getDistance(fileName) == distance:
                allFileNames.append(fileName)
                distance = increaseDistance(distance)
                print(distance)
    return allFileNames


#**********************************************************************************
avgRSSIData = []
scanFolderName = 'No_Obstructions'
arrayAllFiles = getAllFolderFiles(scanFolderName)
#for file in arrayAllFiles:
#    avgRSSIData.append(avgRSSI(getRSSIData(scanFolderName, file)))
#print(avgRSSIData)
print(arrayAllFiles)
