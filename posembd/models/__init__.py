from . import CharBiLSTM.CharBiLSTM
from . import WordBiLSTM
from . import POSTagger

import torch

def createPOSModel(charEmbeddingSize, wordEmbeddingSize, char2id, bilstmSize, datasets):
    charBisltm = CharBILSTM(charEmbeddingSize, wordEmbeddingSize, char2id)
    wordBilstm1 = WordBILSTM(wordEmbeddingSize)
    wordBilstm2 = WordBILSTM(wordEmbeddingSize)

    posModel = POSTagger(charBisltm, wordBilstm1, wordBilstm2, bilstmSize, datasets)
    return posModel

def loadPOSModel(filePath, datasets, char2id):
    fileDict = torch.load(filePath)

    bilstmSize = fileDict['bilstmSize']
    charEmbeddingSize = fileDict['charEmbeddingSize']
    wordEmbeddingSize = fileDict['wordEmbeddingSize']
    stateDict = fileDict['stateDict']

    posModel = createPOSModel(charEmbeddingSize, wordEmbeddingSize, char2id, bilstmSize, datasets)
    posModel.load_state_dict(stateDict)

    return posModel


def savePOSModel(posModel, charEmbeddingSize, wordEmbeddingSize, bilstmSize, filePath):
    fileDict = {
        'bilstmSize': bilstmSize,
        'charEmbeddingSize': charEmbeddingSize,
        'wordEmbeddingSize': wordEmbeddingSize,
        'stateDict': posModel.state_dict()
    }

    torch.save(fileDict, filePath)

    return
