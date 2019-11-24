import torch
from ..io import getDataFromFile

class RawDataset():
    def __init__(self, prefix, trainFile, valFile, testFile, useDelimiters=True):

        self.filePrefix = prefix
        self.files = (trainFile, valFile, testFile)
        self.useDelimiters = useDelimiters

        # Train, val and test data size

    def loadData(self):
        self.data = tuple(getDataFromFile(self.filePrefix + file) for file in self.files)


    def parseData(self):
        rets = tuple(self.__parseData(data, self.useDelimiters) for data in self.data)
        self.data, self.wordCounters = zip(*rets)
        self.sentCounters = tuple(len(data) for data in self.data)


    def extractTagDict(self):
        extracted_tags = {token[1] for sample in self.data[0] for token in sample}
        tags = list(sorted(extracted_tags))

        self.tag2id = {"BOS": 0, "EOS": 1}
        for tag in tags:
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)

        # Criando dicionario para as tags
        self.id2tag = [tag for tag, _ in self.tag2id.items()]


    def extractChars(self):
        return {c for sample in self.data[0] for token in sample for c in token[0]}


    def tensorize(self, char2id):
        rets = tuple(self.__tensorize(data, char2id) for data in self.data)
        self.input, self.target = zip(*rets)
        del self.data

    def __tensorize(self, dataset, char2id):
        inputs = [[torch.LongTensor([char2id.get(c, 1) for c in token[0]]) for token in sample] for sample in dataset]
        targets = [torch.LongTensor([self.tag2id.get(token[1], 0) for token in sample]) for sample in dataset]
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
