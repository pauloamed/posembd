import torch
from torch import nn
from torch.nn.utils import rnn

class POSTagger(nn.Module):

    def __init__(self, charBiLSTM, wordBiLSTM1, wordBiLSTM2, n_bilstm_hidden, datasets):
        super().__init__()

        # Retrieving the model size (#layers and #units)
        self.biLSTMSize = n_bilstm_hidden

        # Retrieving the word emebedding size from the embedding model
        wordEmbeddingSize = charBiLSTM.wordEmbeddingSize

        # Setting the embedding model as the feature extractor
        self.charBILSTM = charBiLSTM
        self.wordBILSTM1 = wordBiLSTM1
        self.wordBILSTM2 = wordBiLSTM2

        # Defining the bilstm layer(s)
        self.tagBiLSTM = nn.LSTM(wordEmbeddingSize, self.biLSTMSize,
                                  1, batch_first=True,
                                  bidirectional=True)

        # Setting the final layer (classifier) for each dataset being used
        classifiers = []
        self.tagSet2classifier = {}

        for d in datasets:
            if d.tagSet not in self.tagSet2classifier:
                self.tagSet2classifier[d.tagSet] = len(classifiers)
                classifiers.append(nn.Linear(self.biLSTMSize * 2, len(d.tag2id)))


        self.classifiers = nn.ModuleList(classifiers)

        # Saving datasets names
        self.dropout = nn.Dropout(0.4)


    def forward(self, inputs, desiredOutput = None, batch_process_char = False):
        # Passing the input through the embeding model in order to retrieve the
        # embeddings

        if desiredOutput is None:
            desiredOutput = [False, False, False, False]

        # Setting output formatting
        output = {
            0 : None, # Context free representations
            1: None, # 1-lvl context representations
            2: None, # 2-lvl context representations
            3: None, # pos refined word embeddings
            "length": None # batch length
        }
        output.update({tagSet: None for tagSet in self.tagSet2classifier})
        embeddings = [None for _ in range(4)]


        embeddings[0], lens = self.charBILSTM.forward(inputs, batch_process = batch_process_char) # Char BiLSTM
        if desiredOutput[0]:
            output[0] = embeddings[0].clone() # Saving output

        embeddings[1], lens = self.wordBILSTM1((embeddings[0], lens)) # 1-Word BiLSTM
        if desiredOutput[1]:
            output[1] = embeddings[1].clone() # Saving output

        embeddings[2], lens = self.wordBILSTM2((embeddings[1], lens))
        if desiredOutput[2]:
            output[2] = embeddings[2].clone() # Saving output

        output["length"] = max(lens) # Saving output

        # Sequence packing
        embeddings[2] = rnn.pack_sequence(embeddings[2], enforce_sorted=False)

        # Passing the embeddings through the bilstm layer(s)
        embeddings[3], _ = self.tagBiLSTM(embeddings[2])
        embeddings[3], _ = rnn.pad_packed_sequence(embeddings[3], batch_first=True)
        if desiredOutput[3]:
            output[3] = embeddings[3].clone()

        # Applying dropout
        embeddings[3] = self.dropout(embeddings[3])

        # Updating view
        # see as: B x L x I (batch_size x length x input_size)
        embeddings[3] = embeddings[3].contiguous().view(-1, output["length"], self.biLSTMSize * 2)

        # Saving final outputs
        # Passing through the final layer for each dataset
        output.update({tagSet : self.classifiers[i](embeddings[3])
                            for tagSet, i in self.tagSet2classifier.items()})

        return output
