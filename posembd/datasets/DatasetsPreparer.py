'''
DatasetsPreparer: classe para gerar usableDatasets, datasets usaveis para treino
'''

class DatasetsPreparer():

    def __init__(dataFolder):
        self.dataFolder

    def prepare(datasets):

        rawDatasets = [
            RawDataset(self.dataFolder, datasets[i]['trainFile'], datasets[i]['valFile'], datasets[i]['testFile'])
            for i in range(len(datasets))
        ]

        for i in range(len(rawDatasets)):
            send_output("\n>> Initializing {} dataset".format(self.name), 1)

            send_output(">>> Started loading dataset", 1)
            rawDatasets[i].loadData()
            send_output("<<< Finished loading dataset", 1)

            send_output(">>> Started parsing data from dataset", 1)
            rawDatasets[i].parseData()
            send_output("<<< Finished parsing data from dataset", 1)

            send_output(">>> Started building tag dict for dataset", 1)
            rawDatasets[i].extractTagDict()
            send_output("<<< Finished building tag dict for dataset", 1)

        send_output("\n>> Building char dict...", 1)
        __buildCharDict(rawDatasets)
        send_output("<< Finished building dicts!", 1)

        for i in range(len(rawDatasets)):
            send_output("\n>> Started preparing {} dataset".format(self.name), 1)
            rawDatasets[i].tensorize()
            send_output("<< Finished preparing {} dataset".format(self.name), 1)
            send_output("<< Finished initializing {} dataset".format(self.name), 1)

        usableDatasets = [
            UsableDataset(rawDatasets[i], datasets[i]['name'], datasets[i]['useTrain'], datasets[i]['usaVal'])
            for i in range(len(datasets))
        ]

        return usableDatasets

    def getChar2idDict():
        return (self.char2id, self.id2char)


    def __buildCharDict(datasets):

        extractedChars = set()
        for dataset in datasets:
            send_output(">>> Started extracting chars from {} dataset".format(self.name), 1)
            extractedChars = extractedChars.union(dataset.extractChars())
            send_output("<<< Finished extracting chars from {} dataset".format(self.name), 1)
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
