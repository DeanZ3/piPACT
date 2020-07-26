import os
import sys
import csv
import numpy as np

#returns array of file names
def getAllFolderFiles(folderName):
    folderPath = os.path.join(r'C:\Users\jzhan\Desktop\piPACT', folderName)
    fileNames = os.listdir(folderPath)
    return sortDistance(fileNames)

def getAllFileUUIDData(folderName, file):
    fileName = os.path.join(folderName, file)
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        lineCount = 0
        data = []
        for row in csv_reader:
            if lineCount != 0:
                data.append(row[3])
            lineCount += 1
        return data

def getAllFileMajorData(folderName, file):
    fileName = os.path.join(folderName, file)
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        lineCount = 0
        data = []
        for row in csv_reader:
            if lineCount != 0:
                data.append(row[4])
            lineCount += 1
        return data

def getAllFileMinorData(folderName, file):
    fileName = os.path.join(folderName, file)
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        lineCount = 0
        data = []
        for row in csv_reader:
            if lineCount != 0:
                data.append(row[5])
            lineCount += 1
        return data

def getAllFileRSSIData(folderName, file):
    fileName = os.path.join(folderName, file)
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        lineCount = 0
        data = []
        for row in csv_reader:
            if lineCount != 0:
                data.append(row[7])
            lineCount += 1
        return data

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

def getAllRSSI(folderName):
    avg = []
    allFiles = getAllFolderFiles(folderName)
    for file in allFiles:
        avg.append(avgRSSI(getRSSIData(folderName, file)))
    if len(getAllDistances(folderName)) == 43:
        avg.remove(avg[0])
    return avg

#returns distance in inches
def getDistance(fileName):
    distance = 0.0
    index = fileName.find('_')
    newName = fileName.replace('_', '', 1)
    distance = newName[index : newName.find('_')]
    return np.float32(distance)

def getAllDistances(folderName):
    distances = []
    for file in getAllFolderFiles(folderName):
        distances.append(getDistance(file))
    #Only take away for log one
    if distances[0] == 0.0:
        distances.remove(0.0)
    return distances

def sortDistance(fileNames):
    copy = fileNames
    orderedFiles = []
    orderedFiles.append(copy[0])
    copy.remove(copy[0])
    while len(copy) > 0:
        if getDistance(copy[0]) < getDistance(orderedFiles[0]):
            orderedFiles.insert(0, copy[0])
        elif getDistance(copy[0]) > getDistance(orderedFiles[len(orderedFiles) - 1]):
            orderedFiles.append(copy[0])
        else:
            for index in range(len(orderedFiles) - 1):
                if getDistance(copy[0]) > getDistance(orderedFiles[index]) and getDistance(copy[0]) < getDistance(orderedFiles[index + 1]):
                    orderedFiles.insert(index + 1, copy[0])
        copy.remove(copy[0])
    return orderedFiles

#returns temperature in fahrenheit
def getTemp(fileName):
    temp = 0
    length = len(fileName)
    temp += int(fileName[length - 25]) * 10
    temp += int(fileName[length - 24])
    return temp

def getAllTemp(folderName):
    temp = []
    allFiles = getAllFolderFiles(folderName)
    for file in allFiles:
        temp.append(getTemp(file))
    temp.remove(temp[0])
    return temp

#returns "relative humidity" in percent form
def getHumidity(fileName):
    humidity = 0
    length = len(fileName)
    humidity += int(fileName[length - 22]) * 10
    humidity += int(fileName[length - 21])
    return humidity

def getAllHumidity(folderName):
    humidity = []
    allFiles = getAllFolderFiles(folderName)
    for file in allFiles:
        humidity.append(getHumidity(file))
    humidity.remove(humidity[0])
    return humidity

def getDate(fileName):
    date = 0
    index = fileName.index("T")
    date = int(fileName[index - 8: index])
    return date

def getAllDate(folderName):
    dates = []
    allFiles = getAllFolderFiles(folderName)
    for file in allFiles:
        dates.append(getDate(file))
    dates.remove(dates[0])
    return dates

#fileName ends in .csv
def getTime(fileName):
    time = 0
    index = fileName.index("T")
    time = int(fileName[index + 1: len(fileName) - 4])
    return time

def getAllTime(folderName):
    times = []
    allFiles = getAllFolderFiles(folderName)
    for file in allFiles:
        times.append(getTime(file))
    times.remove(times[0])
    return times

def tooLong(fileName1, fileName2):
    file1 = getTime(fileName1)
    file2 = getTime(fileName2)
    if int(file1/10000) == int(file2/10000):
        if abs(file1 - file2) <= 1000:
            return True
        else:
            return False
    elif abs(int(file1/10000) - int(file2/10000)) == 1:
        if abs(file1 - file2) <= 5000:
            return True
        else:
            return False
    else:
        return False


#Math functions
def Abs(list):
    newList = []
    for num in list:
        newList.append(abs(num))
    return newList

def ePower(list):
    newList = []
    for num in list:
        newList.append(pow(e, num))
    return newList

def log10(list):
    newList = []
    for num in list:
        newList.append(math.log10(num))
    return newList



#Diagnostic Functions

#Say sick when really they are sick
def truePositive(actual, prediction):
    if actual <= 72.0 and prediction <= 72.0:
        print(prediction, "is a True Positive when distance really is ...", actual)
        return True
    return False

#Say sick when really not sick
def falsePositive(actual, prediction):
    if actual > 72.0 and prediction <= 72.0:
        print("***", end = " ")
        print(prediction, "is a False Positive when distance really is ...", actual)
        return True
    return False

#Say no sick when not sick
def trueNegative(actual, prediction):
    if actual >= 72.0 and prediction > 72.0:
        print(prediction, "is a True Negative when distance really is ...", actual)
        return True
    return False

#Say no sick when really they are sick
def falseNegative(actual, prediction):
    if actual < 72.0 and prediction > 72.0:
        print("***", end = " ")
        print(prediction, "is a False Negative when distance really is ...", actual)
        return True
    return False

def checkALL(actual, prediction):
    trueP = 0
    falseP = 0
    trueN = 0
    falseN = 0
    if truePositive(actual, prediction):
        trueP += 1
    if falsePositive(actual, prediction):
        falseP += 1
    if trueNegative(actual, prediction):
        trueN += 1
    if falseNegative(actual, prediction):
        falseN += 1
    print("\n")
    print(trueP, "True Positives")
    print(falseP, "False Positives")
    print(trueN, "True Negatives")
    print(falseN, "False Negatives")

def writeCSV():
    headers = ['Distance', 'No_Obstructions', 'Short_A', 'Short_S']
    headers += ['2_Shorts', 'Jean_A', 'Jean_S', '2_Jeans', 'Shelf_M', 'Human_A']
    with open('newD.csv', 'w') as csvfile:
        fieldnames = headers
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()

        for index in range(44):
            writer.writerow(csvRow(index))


def csvRow(index):
    content = {}
    content['Distance'] = str(getAllDistances('No_Obstructions')[index])
    content['No_Obstructions'] = str(getAllRSSI('No_Obstructions')[index])
    content['Short_A'] = str(getAllRSSI('Short_A')[index])
    content['Short_S'] = str(getAllRSSI('Short_S')[index])
    content['2_Shorts'] = str(getAllRSSI('2_Shorts')[index])
    content['Jean_A'] = str(getAllRSSI('Jean_A')[index])
    content['Jean_S'] = str(getAllRSSI('Jean_S')[index])
    content['2_Jeans'] = str(getAllRSSI('2_Jeans')[index])
    if index < 39:
        content['Shelf_M'] = str(getAllRSSI('Shelf_M')[index])
    if index < 42:
        content['Human_A'] = str(getAllRSSI('Human_A')[index])
    return content


"""
    fieldNames = ['distance', 'hi']
    theWriter = csv.DictWrtier(f, fieldNames = filedNames)

    theWriter.writeheader()
    for i in range(1, 3):
        theWriter.writerow({'distance' : 'one', 'hi' : 'two'})
"""
"""
print("No Obstructions:", getAllRSSI("No_Obstructions"))
print("\n")
print("2 Jeans:", getAllRSSI("2_Jeans"))
print("\n")
print("2 Shorts:", getAllRSSI("2_Shorts"))
print("\n")
print("Human A:", getAllRSSI("human_A"))
print("\n")
print("Jean A:", getAllRSSI("Jean_A"))
print("\n")
print("Jean S:", getAllRSSI("Jean_S"))
print("\n")
print("Shelf M:", getAllRSSI("Shelf_M"))
print("\n")
print("Short A:", getAllRSSI("Short_A"))
print("\n")
print("Short S:", getAllRSSI("Short_S"))
"""
