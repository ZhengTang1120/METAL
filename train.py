from pyhocon import ConfigFactory
import argparse
from taskManager import TaskManager

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from dataloader import *

from layers import Layers

import torch
from torch import nn

from transformers import BertTokenizer

import onnx
import onnxruntime

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def read_sents(sentences):
    data = {'words': [], 'ners': []}
    for sent in sentences:
        words = ['<CLS>']
        ners = ['<PAD>']
        for row in sent:
            for i, subt in enumerate(tokenizer.tokenize(row.tokens[0])):
                words.append(subt)# use the last sub token
                ners.append(row.tokens[1]) if i == 0 else ners.append('<PAD>')
        words.append('<SEP>')
        ners.append('<PAD>')
        data['words'].append(words)
        data['ners'].append(ners)
    return pd.DataFrame(data)

def get_ids(tokens, key_to_index, unk_id=None):
    return [key_to_index.get(tok, unk_id) for tok in tokens]

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        y = torch.tensor(self.y[index])
        return x, y


parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, help='Filename of the model.', nargs='+')
parser.add_argument('--train', action='store_true', help='Set the code to training purpose.')
parser.add_argument('--config', type=str, help='Filename of the configuration.')
args = parser.parse_args()

config = ConfigFactory.parse_file(args.config)
taskManager = TaskManager(config, 1234)

for task in taskManager.tasks:

    train_df = read_sents(task.trainSentences)

    def get_word_ids(tokens):
        return tokenizer.convert_tokens_to_ids(tokens)
    train_df['word ids'] = train_df['words'].progress_map(get_word_ids)            

    pad_ner = '<PAD>'
    index_to_ner = train_df['ners'].explode().unique().tolist()
    index_to_ner.remove('<PAD>')
    ner_to_index = {t:i for i,t in enumerate(index_to_ner)}
    ner_to_index['<PAD>'] = -100
    pad_ner_id = ner_to_index[pad_ner]
    def get_ner_ids(ners):
        return get_ids(ners, ner_to_index)
    train_df['ner ids'] = train_df['ners'].progress_map(get_ner_ids)

    dev_df = read_sents(task.devSentences)
    dev_df['word ids'] = dev_df['words'].progress_map(get_word_ids)
    dev_df['ner ids'] = dev_df['ners'].progress_map(lambda x: get_ids(x, ner_to_index))

def collate_fn(batch):
    # separate xs and ys
    xs, ys = zip(*batch)
    # get lengths
    lengths = [len(x) for x in xs]
    # pad sequences
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=pad_ner_id)
    # return padded
    return x_padded, y_padded, lengths

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# hyperparameters
lr = 1e-5
weight_decay = 1e-5
batch_size = 100
shuffle = True
n_epochs = 10
hidden_size = 100
num_layers = 2
bidirectional = True
dropout = 0.1
output_size = len(index_to_ner)

# initialize the model, loss function, optimizer, and data-loader
model = Layers(config, output_size)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
train_ds = MyDataset(train_df['word ids'], train_df['ner ids'])
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
dev_ds = MyDataset(dev_df['word ids'], dev_df['ner ids'])
dev_dl = DataLoader(dev_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
best = 0

if args.train:
    # train the model
    for epoch in range(n_epochs):
        losses, acc = [], []
        model.train()
        for x_padded, y_padded, _ in tqdm(train_dl, desc=f'epoch {epoch+1} (train)'):
            # clear gradients
            model.zero_grad()
            # send batch to right device
            x_padded = x_padded
            y_padded = y_padded
            # predict label scores
            y_pred = model(x_padded)
            # reshape output
            y_true = torch.flatten(y_padded)
            y_pred = y_pred.view(-1, output_size)
            mask = y_true != pad_ner_id
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            # compute loss
            loss = loss_func(y_pred, y_true)
            # accumulate for plotting
            gold = y_true.detach().numpy()
            pred = np.argmax(y_pred.detach().numpy(), axis=1)
            losses.append(loss.detach().item())
            acc.append(accuracy_score(gold, pred))
            # backpropagate
            loss.backward()
            # optimize model parameters
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            losses, acc = [], []
            golds = []
            preds = []
            for x_padded, y_padded, _ in tqdm(dev_dl, desc=f'epoch {epoch+1} (dev)'):
                x_padded = x_padded
                y_padded = y_padded
                y_pred = model(x_padded)
                y_true = torch.flatten(y_padded)
                y_pred = y_pred.view(-1, output_size)
                mask = y_true != pad_ner_id
                y_true = y_true[mask]
                y_pred = y_pred[mask]
                loss = loss_func(y_pred, y_true)
                gold = y_true.cpu().numpy().tolist()
                pred = np.argmax(y_pred.cpu().numpy(), axis=1).tolist()
                losses.append(loss.cpu().item())
                golds += gold
                preds += pred
                acc.append(accuracy_score(gold, pred))
            loss, acc, f1 = np.mean(losses), np.mean(acc), f1_score(np.array(golds), np.array(preds), labels=[i for i, l in enumerate(index_to_ner) if l!='O' and l!='<PAD>'], average='micro')
            print (epoch, f1)
            if f1 > best:
                torch.save(model.state_dict(), "best_model.pt")
                input_names = ["words"]
                output_names = ["ners"]
                torch.onnx.export(model,
                    x_padded,
                    "best_model.onnx",
                    export_params=True,
                    do_constant_folding=True,
                    input_names = input_names,
                    output_names = output_names,
                    opset_version=10, 
                    dynamic_axes = {"words": {1: 'sent length'}})
                best = f1
else:
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    onnx_model = onnx.load("best_model.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("best_model.onnx")
    with torch.no_grad():
        losses, acc = [], []
        golds = []
        preds = []
        for x_padded, y_padded, _ in tqdm(dev_dl, desc=f'dev eval'):
            x_padded = x_padded
            y_padded = y_padded
            y_pred_o = model(x_padded)
            y_true = torch.flatten(y_padded)
            y_pred = y_pred_o.view(-1, output_size)
            mask = y_true != pad_ner_id
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            loss = loss_func(y_pred, y_true)
            gold = y_true.cpu().numpy().tolist()
            pred = np.argmax(y_pred.cpu().numpy(), axis=1).tolist()
            losses.append(loss.cpu().item())
            golds += gold
            preds += pred
            acc.append(accuracy_score(gold, pred))

            ort_inputs = {ort_session.get_inputs()[i].name: to_numpy(x) for i, x in enumerate([x_padded])}
            ort_outs = ort_session.run(None, ort_inputs)

            np.testing.assert_allclose(y_pred_o.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

        loss, acc, f1 = np.mean(losses), np.mean(acc), f1_score(np.array(golds), np.array(preds), labels=[i for i, l in enumerate(index_to_ner) if l!='O' and l!='<PAD>'], average='micro')

        print (f1)



