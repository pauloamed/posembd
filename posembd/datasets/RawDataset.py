import torch
from ..io import getDataFromFile

from tqdm import tqdm

class RawDataset():
    def __init__(self, name, prefix, trainFile, valFile, testFile, tagSet, useDelimiters=True):

        self.name = name
        self.filePrefix = prefix
        self.files = (trainFile, valFile, testFile)
        self.useDelimiters = useDelimiters
        self.tagSet = tagSet

        # Train, val and test data size

    def loadData(self):
        self.data = (getDataFromFile(self.filePrefix + file) for file in tqdm(self.files, "Loading files"))


    def parseData(self):
        rets = tuple(self.__parseData(data, self.useDelimiters)
                for data in tqdm(self.data, "Parsing {} data".format(self.name)))
        self.data, self.wordCounters = zip(*rets)
        self.sentCounters = tuple(len(data) for data in self.data)


    def extractTagDict(self):
        extracted_tags = {token[1] for sample in tqdm(self.data[0], "Extracting tags from {}".format(self.name))
                for token in sample}
        tags = list(sorted(extracted_tags))

        tag2id = {'BOS': 0, 'EOS': 1}
        for tag in tags:
            if tag not in tag2id:
                tag2id[tag] = len(tag2id)

        return tag2id


    def extractChars(self):
        return {c for sample in tqdm(self.data[0], "Extracting chars from {}".format(self.name))
                    for token in sample for c in token[0]}


    def tensorize(self, char2id):
        rets = tuple(self.__tensorize(data, char2id)
                for data in tqdm(self.data, "Tensorizing {} dataset".format(self.name)))
        self.inputData, self.targetData = zip(*rets)
        del self.data

    def __tensorize(self, dataset, char2id):
        inputs = [[torch.LongTensor([char2id.get(c, 1) for c in token[0]]) for token in sample]
                    for sample in dataset]
        targets = [torch.LongTensor([self.tag2id.get(token[1], 0) for token in sample])
                    for sample in dataset]
        return (inputs, targets)

    def __parseData(self, data, use_delimiters):
        counter = 0

        BOW = "\002" if use_delimiters else ""
        EOW = "\003" if use_delimiters else ""
        BOS = [["\001", "BOS"]] if use_delimiters else []
        EOS = [["\004", "EOS"]] if use_delimiters else []

        dataset = []
        for sample in data.split('\n'):
            if sample == '':
                continue

            s = sample.strip().split(' ')
            counter += len(s)
            middle = [[BOW + token.rsplit('_', 1)[0] + EOW, token.rsplit('_', 1)[1]]
                                                                    for token in s]
            dataset.append(BOS + middle + EOS)
        return dataset, counter
