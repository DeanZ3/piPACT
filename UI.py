from algorithms import *

fakeIndexCase = getAllFolderFiles("No_Obstructions")
fakeIndexCaseUUID = "a643ae8a-bd51-11ea-90ef-dca63233b0c1"
fakeIndexCaseMajor = "1"
fakeIndexCasaeMinor = "1"

print("Hi, Welcome to Dean's Contact Tracing 'app'!")
print("\n")

folderName = input("Enter name of folder where you have stored your bluetooth chirps: ")
allFiles = getAllFolderFiles(folderName)

def checkID(testFile, illUUID, illMajor, illMinor):
    indexUUID = []
    indexMajor = []
    indexMinor = []
    finalIndexCheck = []
    for UUID in getAllFileUUIDData(folderName, testFile):
        print(UUID)
        if UUID == illUUID:
            print(getAllFileUUIDData(folderName, testFile).index(UUID))
            indexUUID.append(getAllFileUUIDData(folderName, testFile).index(UUID))
    for major in getAllFileMajorData(folderName, testFile):
        if major == illMajor:
            indexMajor.append(getAllFileMajorData(folderName, testFile).index(major))
    for minor in getAllFileMinorData(folderName, testFile):
        if minor == illMinor:
            indexMinor.append(getAllFileMinorData(folderName, testFile).index(minor))
    #print(indexUUID)
    #print(indexMajor)
    #print(indexMinor)
    for index in indexUUID:
        if index in indexMajor and index in indexMinor:
            finalIndexCheck.append(index)
    return finalIndexCheck

match = {}
for file in allFiles:
    match[file] = checkID(file, fakeIndexCaseUUID, fakeIndexCaseMajor, fakeIndexCasaeMinor)

print(match)
