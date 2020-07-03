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
    allFileNames = []
    folderPath = os.path.join(r'C:\Users\jzhan\Desktop\piPACT', folderName)
    fileNames = os.listdir(folderPath)
    for fileName in fileNames:
        allFileNames.append(fileName)
    return allFileNames


#**********************************************************************************
avgRSSIData = []
scanFolderName = 'No_Obstructions'
arrayAllFiles = getAllFolderFiles(scanFolderName)
for file in arrayAllFiles:
    avgRSSIData.append(avgRSSI(getRSSIData(scanFolderName, file)))
print(avgRSSIData)

otherData = []
folderName = '1_Short_on_A'
allFiles = getAllFolderFiles(folderName)
for file in allFiles:
    otherData.append(avgRSSI(getRSSIData(folderName, file)))
print(otherData)

hi = []
lol = '1_Short_on_S'
he = getAllFolderFiles(lol)
for file in he:
    hi.append(avgRSSI(getRSSIData(lol, file)))
print(hi)
