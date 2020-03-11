from ..globals import getLogLvl, OUTPUT_PATH

def saveToPickle(filePath, obj):
    pickleOut = open(filePath, "wb")
    pickle.dump(obj, pickleOut)
    pickleOut.close()


def loadFromPickle(filePath):
    pickleIn = open(filePath, "rb")
    obj = pickle.load(pickleIn)
    pickleIn.close()

    return obj


def saveDictToFile(filePath, fileDict):
    saveToPickle(filePath, fileDict)


def getDictFromFile(filePath):
    return loadFromPickle(filePath)


def sendOutput(str, log_level):
    if log_level <= getLogLvl():
        print(str)
    try:
        file = open(OUTPUT_PATH, "a")
        file.write(str + "\n")
        file.close()
    except:
        if log_level <= getLogLvl():
            print("Was not able to open and write on output file")


def getDataFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as f:
        data = f.read()
    return data
