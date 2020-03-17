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

        self.id2tag = [None for _ in range(len(self.tag2id))]
        for tag, id in self.tag2id.items():
            self.id2tag[id] = tag


    def __str__(self):
        ret = ""
        ret += ("=================================================================\n")
        ret += ("{} Dataset\n".format(self.name))
        ret += ("Train dataset #sents: {} #words: {}\n".format(len(self.trainInput), self.wordCountTrain))
        ret += ("Val dataset #sents: {} #words: {}\n".format(len(self.valInput), self.wordCountVal))
        ret += ("Test dataset #sents: {} #words: {}\n".format(len(self.testInput),self.wordCountTest))
        ret += ("Tagset: [{}]\n".format(", ".join(self.id2tag)))
        ret += ("=================================================================\n")

        return ret
