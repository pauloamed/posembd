'''
DatasetsPreparer: classe para gerar usableDatasets, datasets usaveis para treino
'''

from .RawDataset import RawDataset
from ..io import sendOutput

class DatasetsPreparer():

    def __init__(self, dataFolder):
        self.dataFolder = dataFolder

    def prepare(self, datasets):

        rawDatasets = [
            RawDataset(self.dataFolder, datasets[i]['trainFile'], datasets[i]['valFile'], datasets[i]['testFile'])
            for i in range(len(datasets))
        ]

        for i in range(len(rawDatasets)):
            sendOutput("\n>> Initializing {} dataset".format(rawDatasets[i].name), 1)

            sendOutput(">>> Started loading dataset", 1)
            rawDatasets[i].loadData()
            sendOutput("<<< Finished loading dataset", 1)

            sendOutput(">>> Started parsing data from dataset", 1)
            rawDatasets[i].parseData()
            sendOutput("<<< Finished parsing data from dataset", 1)

            sendOutput(">>> Started building tag dict for dataset", 1)
            rawDatasets[i].extractTagDict()
            sendOutput("<<< Finished building tag dict for dataset", 1)

        sendOutput("\n>> Building char dict...", 1)
        self.__buildCharDict(rawDatasets)
        sendOutput("<< Finished building dicts!", 1)

        for i in range(len(rawDatasets)):
            sendOutput("\n>> Started preparing {} dataset".format(rawDatasets[i].name), 1)
            rawDatasets[i].tensorize()
            sendOutput("<< Finished preparing {} dataset".format(rawDatasets[i].name), 1)
            sendOutput("<< Finished initializing {} dataset".format(rawDatasets[i].name), 1)

        usableDatasets = [
            UsableDataset(rawDatasets[i], datasets[i]['name'], datasets[i]['useTrain'], datasets[i]['usaVal'])
            for i in range(len(datasets))
        ]

        return usableDatasets

    def getDicts(self):
        return (self.char2id, self.id2char)


    def __buildCharDict(self, datasets):

        extractedChars = set()
        for dataset in datasets:
            sendOutput(">>> Started extracting chars from {} dataset".format(self.name), 1)
            extractedChars = extractedChars.union(dataset.extractChars())
            sendOutput("<<< Finished extracting chars from {} dataset".format(self.name), 1)
        chars = [' ', 'UNK'] + list(sorted(extractedChars))

        # Criando estruturas do vocabulário
        self.char2id = {char: index for index, char in enumerate(chars)}
        self.id2char = [char for char, _ in char2id.items()]



    # def loadDatasets(dataFolder, datasetsFiles):
    #     pf = dataFolder
    #     datasets = [
    #         Dataset(pf + dataset['trainFile'], pf + dataset['valFile'], pf + dataset['testFile'], dataset['name'], use_train=dataset['trainFlag'],
    #                         use_val=dataset['valFlag'])
    #         for dataset in datasets
    #     ]
    #
    #     return datasets
