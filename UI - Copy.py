from algorithms import *

#Set values to test
fakeIndexCase = getAllFolderFiles("No_Obstructions")
fakeIndexCaseUUID = "a643ae8a-bd51-11ea-90ef-dca63233b0c1"
fakeIndexCaseMajor = "1"
fakeIndexCasaeMinor = "1"

#Intro/input
print("Hi, Welcome to Dean's Contact Tracing 'app'!")
print("\n")
folderName = input("Enter name of folder where you have stored your bluetooth chirps: ")
allFiles = getAllFolderFiles(folderName)

#Check ID to see whether or not the device that released signal = sick person's device
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



#TODO -> Error check and make sure it runs correctly...

#Address of pis change every once in a while, so need to change fakeID stuff
match = {}
for file in allFiles:
    match[file] = checkID(file, fakeIndexCaseUUID, fakeIndexCaseMajor, fakeIndexCasaeMinor)

#print(match)

#Function that tests the final part in contact tracing
def checkRequirements(match)
    tooClose = False
    for file1 in match:
        for file2 in match:
            if tooLong(file1, file2):
                if calculateDistance(folderName, file1, file2, "O"):
                    print("You have been close to someone with the virus for too long")
    print("You have not been too close to someone with the virus for too long")
