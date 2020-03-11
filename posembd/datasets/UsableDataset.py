import torch

class UsableDataset():
    def __init__(self, rawDataset, useTrain=True, useVal=True):
        self.name = rawDataset.name
        self.tagSet = rawDataset.tagSet

        # Setting bool flags
        self.useTrain = useTrain
        self.useVal = useVal

        # Setting data
        self.trainInput, self.valInput, self.testInput = rawDataset.inputData
        self.trainTarget, self.valTarget, self.testTarget = rawDataset.targetData

        # Setting sizes
        self.wordCountTrain, self.wordCountVal, self.wordCountTest = rawDataset.wordCounters
        self.sentCountTrain, self.sentCountVal, self.sentCountTest = rawDataset.sentCounters

        self.tag2id = rawDataset.tag2id


    def __str__(self):
        ret = ""
        ret += ("=================================================================\n")
        ret += ("{} Dataset\n".format(self.name))
        ret += ("Train dataset #sents: {} #words: {}\n".format(len(self.trainInput), self.wordCountTrain))
        ret += ("Val dataset #sents: {} #words: {}\n".format(len(self.valInput), self.wordCountVal))
        ret += ("Test dataset #sents: {} #words: {}\n".format(len(self.testInput),self.wordCountTest))
        ret += ("Tag set: [{}]\n".format(", ".join(self.id2tag)))
        ret += ("=================================================================\n")

        return ret
