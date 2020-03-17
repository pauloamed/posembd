from .CharBiLSTM import CharBiLSTM
from .WordBiLSTM import WordBiLSTM
from .POSTagger import POSTagger

import torch

def createPOSModel(charEmbeddingSize, wordEmbeddingSize, posEmbeddingSize, char2id, datasets):
    charBiLSTM = CharBiLSTM(charEmbeddingSize, wordEmbeddingSize, char2id)
    wordBilstm1 = WordBiLSTM(wordEmbeddingSize)
    wordBilstm2 = WordBiLSTM(wordEmbeddingSize)

    posModel = POSTagger(charBisltm, wordBilstm1, wordBilstm2, posEmbeddingSize, datasets)
    return posModel

def loadPOSModel(filePath, datasets, char2id):
    fileDict = torch.load(filePath)

    bilstmSize = fileDict['posEmbeddingSize']
    charEmbeddingSize = fileDict['charEmbeddingSize']
    wordEmbeddingSize = fileDict['wordEmbeddingSize']
    stateDict = fileDict['stateDict']

    posModel = createPOSModel(charEmbeddingSize, wordEmbeddingSize, char2id, posEmbeddingSize, datasets)
    posModel.load_state_dict(stateDict)

    return posModel


def savePOSModel(posModel, charEmbeddingSize, wordEmbeddingSize, posEmbeddingSize, filePath):
    fileDict = {
        'posEmbeddingSize': posEmbeddingSize,
        'charEmbeddingSize': charEmbeddingSize,
        'wordEmbeddingSize': wordEmbeddingSize,
        'stateDict': posModel.state_dict()
    }

    torch.save(fileDict, filePath)

    return
