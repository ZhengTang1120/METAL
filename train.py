from pyhocon import ConfigFactory
import argparse
from taskManager import TaskManager

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from gensim.models import KeyedVectors

from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from dataloader import *

from layers import Layers

def read_sents(sentences):
    data = {'words': [], 'ners': []}
    for sent in sentences:
        words = list()
        ners = list()
        for row in sent:
            words.append(row.tokens[0])
            ners.append(row.tokens[1])
        data['words'].append(words)
        data['ners'].append(ners)
    return pd.DataFrame(data)

def get_ids(tokens, key_to_index, unk_id=None):
    return [key_to_index.get(tok, unk_id) for tok in tokens]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help='Filename of the model.', nargs='+')
    parser.add_argument('--train', action='store_true', help='Set the code to training purpose.')
    parser.add_argument('--config', type=str, help='Filename of the configuration.')
    args = parser.parse_args()

    if args.train:
        config = ConfigFactory.parse_file(args.config)
        taskManager = TaskManager(config, 1234)

        glove = KeyedVectors.load_word2vec_format('/data1/home/zheng/new/processors/main/src/main/python/glove.840B.300d.10f.txt')
        pad_tok = '<pad>'
        pad_emb = np.zeros(300)
        glove.add_vector(pad_tok, pad_emb)
        pad_tok_id = glove.key_to_index[pad_tok]
        unk_tok = '<unk>'
        unk_emb = np.random.rand(300)
        glove.add_vector(unk_tok, unk_emb)
        unk_id = glove.key_to_index[unk_tok]
        for task in taskManager.tasks:

            train_df = read_sents(task.trainSentences)

            def get_word_ids(tokens):
                return get_ids(tokens, glove.key_to_index, unk_id)
            train_df['word ids'] = train_df['words'].progress_map(get_word_ids)
            train_df
            

            pad_ner = '<pad>'
            index_to_ner = train_df['ners'].explode().unique().tolist() + [pad_ner]
            ner_to_index = {t:i for i,t in enumerate(index_to_ner)}
            pad_ner_id = ner_to_index[pad_ner]
            def get_ner_ids(ners):
                return get_ids(ners, ner_to_index)
            train_df['ner ids'] = train_df['ners'].progress_map(get_ner_ids)

            dev_df = read_sents(task.devSentences)
            dev_df['word ids'] = dev_df['words'].progress_map(lambda x: get_ids(x, glove.key_to_index, unk_id))
            dev_df['ner ids'] = dev_df['ners'].progress_map(lambda x: get_ids(x, ner_to_index))

# hyperparameters
lr = 1e-3
weight_decay = 1e-5
batch_size = 100
shuffle = True
n_epochs = 10
vectors = glove.vectors
hidden_size = 100
num_layers = 2
bidirectional = True
dropout = 0.1
output_size = len(index_to_ner)
word_size = vectors.shape[0]

# initialize the model, loss function, optimizer, and data-loader
model = Layers(config, word_size, output_size, vectors)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
train_ds = MyDataset(train_df['word ids'], train_df['ner ids'])
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
dev_ds = MyDataset(dev_df['word ids'], dev_df['ner ids'])
dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

train_loss, train_acc = [], []
dev_loss, dev_acc = [], []





