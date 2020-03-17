import torch
from torch import nn
import torch.nn.utils.rnn as rnn

class CharBiLSTM(nn.Module):
    def __init__(self, charEmbeddingSize, wordEmbeddingSize, char2id):
        super().__init__()

        # Setting the embeddings dimensions
        self.charEmbeddingSize = charEmbeddingSize
        self.wordEmbeddingSize = wordEmbeddingSize

        # Setting the char embedding lookup table
        self.char_embeddings_table = nn.Embedding(len(char2id),
                                                  charEmbeddingSize,
                                                  padding_idx=0)

        # Setting up the first BILSTM (char-level/morpho)
        self.bilstm = nn.LSTM(charEmbeddingSize,
                              wordEmbeddingSize, 1,
                              batch_first=True,
                              bidirectional=True)

        # Setting up the projection layers
        self.projectionLayer = nn.Linear(2*wordEmbeddingSize, wordEmbeddingSize)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, batch_process):
        # For each sample (sentence) on the batch, get a list of 1st-level-word embeddings
        outputs, lens = [], []
        for sample in inputs:

            # Retrieving char embeddings for each word in sample
            embeddedSample = [self.dropout(self.char_embeddings_table(word)) for word in sample]

            if batch_process:
                # Sequence packing
                packedEmbeddedSample = rnn.pack_sequence(embeddedSample, enforce_sorted=False)

                # Passes the words on the sentence altogether through the char_bilstm
                output, _ = self.bilstm(packedEmbeddedSample)

                # Sequence unpacking
                padded_output, output_lens = rnn.pad_packed_sequence(output, batch_first=True)

                # For each word on the sample, save two outputs from the BILSTM outputs
                # Saving output of reverse lstm (output on 0)
                outputReverse = padded_output[:,0,self.wordEmbeddingSize:]

                # Saving output of forward lstm (output on LENGTH-1)
                outputForward = [padded_output[i, word_len-1, :self.wordEmbeddingSize]
                                               for i, word_len in enumerate(output_lens)]
                outputForward = torch.stack(outputForward)

                # Concat outputs
                concatEmbeddings = torch.cat((outputReverse, outputForward), dim=1)

                # Projects on a word_embedding dimension
                wordEmbeddings = self.projectionLayer(concatEmbeddings)

            else:
                # Sequence packing
                wes = []
                for i in range(len(sample)):
                    output, _ = self.bilstm(embeddedSample[i].unsqueeze(0))

                    outputReverse = output[0, 0, self.wordEmbeddingSize:]
                    outputForward = output[0, len(embeddedSample[i])-1,:self.wordEmbeddingSize]

                    # Concat outputs
                    concatEmbeddings = torch.cat((outputReverse, outputForward), dim=0)

                    # Projects on a word_embedding dimension
                    we = self.projectionLayer(concatEmbeddings)
                    wes.append(we)

                wordEmbeddings = torch.stack(wes)


            # Saves to output
            outputs.append(self.dropout(wordEmbeddings))
            lens.append(len(sample))

        # Padding sequence as needed
        outputs = rnn.pad_sequence(outputs, batch_first=True)

        return outputs, lens
