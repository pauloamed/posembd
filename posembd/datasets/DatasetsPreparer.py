'''
DatasetsPreparer: classe para gerar usableDatasets, datasets usaveis para treino
'''

from .RawDataset import RawDataset
from .UsableDataset import UsableDataset
from ..io import sendOutput

from tqdm import tqdm

class DatasetsPreparer():

    def __init__(self, dataFolder):
        self.dataFolder = dataFolder

    def prepare(self, datasets):

        rawDatasets = [
            RawDataset(datasets[i]['name'], self.dataFolder, datasets[i]['trainFile'], datasets[i]['valFile'], datasets[i]['testFile'], datasets[i]['tagSet'])
            for i in range(len(datasets))
        ]

        # pode ser paralelizado
        for i in range(len(rawDatasets)):
            rawDatasets[i].loadData()
            rawDatasets[i].parseData()

        tagSet2tagDict = dict()
        for i in range(len(rawDatasets)):
            tagDict = rawDatasets[i].extractTagDict()

            if rawDatasets[i].tagSet not in tagSet2tagDict:
                tagSet2tagDict[rawDatasets[i].tagSet] = tagDict
            else:
                tagSet2tagDict[rawDatasets[i].tagSet].update(tagDict)

        for i in range(len(rawDatasets)):
            rawDatasets[i].tag2id = tagSet2tagDict[rawDatasets[i].tagSet]

        self.__buildCharDict(rawDatasets)

        # pode ser paralelizado
        for i in range(len(rawDatasets)):
            rawDatasets[i].tensorize(self.char2id)

        usableDatasets = [
            UsableDataset(rawDatasets[i], datasets[i]['useTrain'], datasets[i]['useVal'])
            for i in range(len(datasets))
        ]

        return usableDatasets


    def getDicts(self):
        return (self.char2id, self.id2char)


    def __buildCharDict(self, datasets):

        extractedChars = set()
        for dataset in datasets:
            extractedChars = extractedChars.union(dataset.extractChars())
        chars = [' ', 'UNK'] + list(sorted(extractedChars))

        # Criando estruturas do vocabulário
        self.char2id = {char: index for index, char in tqdm(enumerate(chars), "Setting char2id dict", total=len(chars), file=sys.stdout)}
        self.id2char = [char for char, _ in tqdm(self.char2id.items(), "Setting id2char list", file=sys.stdout)]
