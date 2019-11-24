import torch

class UsableDataset():
    def __init__(self, name, rawDataset, useTrain=True, useVal=True):
        self.name = name

        # Setting bool flags
        self.use_train = use_train
        self.use_val = use_val

        # Train, val and test data size
        self.sent_train_size = len(self.train_data[0])
        self.sent_val_size = len(self.val_data[0])
        self.sent_test_size = len(self.test_data[0])

        # Setting training and val loss
        self.train_loss = 0.0
        self.val_loss = 0.0

        # Setting test counters
        self.class_correct = [0 for _ in range(len(self.tag2id))]
        self.class_total = [0 for _ in range(len(self.tag2id))]


    def __str__(self):
        ret = ""
        ret += ("=================================================================\n")
        ret += ("{} Dataset\n".format(self.name))
        ret += ("Train dataset #sents: {} #words: {}\n".format(len(self.train_input), self.word_train_size))
        ret += ("Val dataset #sents: {} #words: {}\n".format(len(self.val_input), self.word_val_size))
        ret += ("Test dataset #sents: {} #words: {}\n".format(len(self.test_input),self.word_test_size))
        ret += ("Tag set: [{}]\n".format(", ".join(self.id2tag)))
        ret += ("=================================================================\n")

        return ret
