import numpy as np
from collections import defaultdict
import os
import tensorflow as tf
from nltk.tokenize import word_tokenize

class DataSet:
    def __init__(self, path, token_map, label2idx=None, sequence_len=100):
        self.id = os.path.basename(path).replace(".train", "").replace(".dev", "")
        self.word_map = token_map
        self.sequence_len = sequence_len

        self.data, self.label = self.load_text(path)
        # learn mapping
        if not label2idx:
            self.label2idx = dict(self.learn_label_encoding(self.label))
        else:
            self.label2idx = label2idx
        self.idx2label = {idx: lab for lab, idx in self.label2idx.items()}
        # store the encoded version of everything
        self.data = self.tokenize(self.data)
        self.encoded_data, self.encoded_label = self.encode_text_and_label(self.data, self.label)
        self.elabel2edata = self.learnlabel2datamap(self.encoded_data, self.encoded_label)

    def load_text(self, path):
        data = list()
        label = list()
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip().split("\t")
                if len(line) > 1:
                    label.append(line[1])
                else:
                    label.append("UNCONFIDENT_FROM_SLAD")
                data.append(line[0].lower())

        return data, label

    def tokenize(self, data, tokenizer= word_tokenize):#lambda x: x.split(" ")):
        data = list(map(tokenizer, data))
        return data

    def learn_label_encoding(self, label):
        label_map = defaultdict(int)
        label_map.default_factory = label_map.__len__
        for element in set(label):
            label_map[element]
        return label_map

    def encode_text_and_label(self, data, label):
        encoded_text = list()
        encoded_label = list()
        for tokens, lab in zip(data, label):
            if lab not in self.label2idx:
                continue
            encoded_label.append(self.label2idx[lab])
            encoded_text.append(np.array([self.word_map.get(tokens[i], self.word_map["<unk>"]) if len(tokens) > i else self.word_map["<pad>"] for i in range(self.sequence_len)], dtype=np.int))
        return np.array(encoded_text, dtype=np.int), np.array(encoded_label, dtype=np.int)

    def learnlabel2datamap(self, encoded_text, encoded_label):
        mapping = dict()
        for idx in self.idx2label:
            mapping[idx] = encoded_text[encoded_label == idx]
        return mapping

    def __repr__(self):
        return self.id

    def generate(self, batch_size=10):
        l = len(self.encoded_data)
        for idx in range(0, l, batch_size):
            end = min(idx + batch_size, l)
            yield self.encoded_data[idx:end], self.encoded_label[idx:end]

    def shuffle(self):
        seed = np.random.get_state()
        np.random.set_state(seed)
        np.random.shuffle(self.encoded_data)
        np.random.set_state(seed)
        np.random.shuffle(self.encoded_label)
