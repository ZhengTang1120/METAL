import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel, BertConfig

DEFAULT_DROPOUT_PROB: float = 0.1
DEFAULT_LEARNED_WORD_EMBEDDING_SIZE: int = 128
DEFAULT_CHAR_EMBEDDING_SIZE: int = 32
DEFAULT_CHAR_RNN_STATE_SIZE: int = 16
DEFAULT_POS_TAG_EMBEDDING_SIZE: int = 0 # no POS tag embeddings by default
DEFAULT_NE_TAG_EMBEDDING_SIZE: int = 0 # no NE tag embeddings by default
DEFAULT_DISTANCE_EMBEDDING_SIZE: int = 0 # no distance embeddings by default
DEFAULT_POSITION_EMBEDDING_SIZE: int = 0 # no position embeddings by default
DEFAULT_DISTANCE_WINDOW_SIZE: int = 0
DEFAULT_USE_IS_PREDICATE: int = 0
MAX_INTERMEDIATE_LAYERS: int = 10
nonlin_map = {"relu":"relu", "tanh":"tanh", "":""}

class Layers(nn.Module):
    def __init__(self, config, output_size, postag_size=0, ner_size=0):
        super().__init__()

        paramPrefix = "mtl.layers"

        self.initialLayer = BERTLayer("bert-base-uncased")
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        input_size = bert_config.hidden_size

        i = 1
        nonlinAsString = config.get_string(f"mtl.task{i}.layers" + ".final" + ".nonlinearity", "")
        if nonlinAsString in nonlin_map:
            nonlin = nonlin_map[nonlinAsString]
        else:
            raise RuntimeError(f"ERROR: unknown non-linearity {nonlinAsString}!")
        self.finalLayer = ForwardLayer(input_size, output_size, nonlin)

    def forward(self, states, lengths, headPositions=None):

        if self.initialLayer is not None:
            states = self.initialLayer(states)
        if self.finalLayer is not None:
            states = self.finalLayer(states, headPositions)

        return states



class InitialLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, words):
        raise NotImplementedError

class IntermediateLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputExpressions):
        raise NotImplementedError

class FinalLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputExpressions, modifierHeadPairs):
        raise NotImplementedError

class BERTLayer(InitialLayer):
    def __init__(self, model_name):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, words, segment_ids=None, masks=None):
        output = self.model(words)
        h = output.last_hidden_state

        return h


class EmbeddingLayer(InitialLayer):
    def __init__(self, embeddings, word_size, learnedWordEmbeddingSize, dropout, 
                tag_size, posTagEmbeddingSize, ner_size, neTagEmbeddingSize, 
                distanceWindowSize, distanceEmbeddingSize, positionEmbeddingSize,
                useIsPredicate):
        super().__init__()
        self.useIsPredicate = useIsPredicate
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings)
        self.pr_embedding = nn.Embedding.from_pretrained(embeddings=embeddings)
        self.wd_embedding = nn.Embedding(word_size, learnedWordEmbeddingSize)
        # self.ch_embedding = nn.Embedding(char_size, charEmbeddingSize)
        # self.ch_lstm      = nn.LSTM(input_size=charEmbeddingSize, hidden_size=char_hidden_size, 
        #                             num_layers=1, bidirectional=True, dropout=dropout)
        if tag_size > 0 and posTagEmbeddingSize > 0:
            self.tg_embedding = nn.Embedding(tag_size, posTagEmbeddingSize)
        if ner_size > 0 and neTagEmbeddingSize > 0:
            self.ne_embedding = nn.Embedding(ner_size, neTagEmbeddingSize)
        if distanceWindowSize > 0 and distanceEmbeddingSize > 0:
            self.dw_embedding = nn.Embedding(distanceWindowSize * 2 + 3, distanceEmbeddingSize)
        if positionEmbeddingSize > 0:
            self.ps_embedding = nn.Embedding(101, positionEmbeddingSize)
        else:
            self.ps_embedding = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, words, tags=None, nes=None, headPositions=None):
        word_embeds1 = self.pr_embedding(words)
        word_embeds2 = self.wd_embedding(words)

        tag_embed = self.tg_embedding(tags) if tags and self.tg_embedding else None
        ner_embed = self.ne_embedding(nes)  if nes  and self.ne_embedding else None

        if headPositions and self.useIsPredicate:
            pred_embed = torch.FloatTensor([1 if i==predicatePosition else 0 for i, predicatePosition in enumerate(headPositions)]).unsqueeze(1)
        else:
            pred_embed = None

        if headPositions and self.dw_embedding:
            dists = [max(i-predicatePosition+self.distanceWindowSize+1, 0) if i-predicatePosition <= self.distanceWindowSize else 2 * self.distanceWindowSize + 2 for i, predicatePosition in enumerate(headPositions)]
            dist_embed = self.dw_embedding(torch.LongTensor(dists))
        else:
            dist_embed = None

        if self.ps_embedding:
            values = [i if i<100 else 100 for i, word in enumerate(words)]
            pos_embed = self.ps_embedding(torch.LongTensor(values))
        else:
            pos_embed = None

        embedParts = [word_embeds1, word_embeds2, tag_embed, ner_embed, dist_embed, pos_embed, pred_embed]
        embedParts = [ep for ep in embedParts if ep is not None]
        embed = torch.cat(embedParts, dim=2)
        return self.dropout(embed)

class ForwardLayer(FinalLayer):
    def __init__(self, input_size, output_size, nonlinearity):
        super().__init__()
        self.classifier = nn.Linear(input_size, output_size) 
        if nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        else:
            self.nonlinearity = None
    def forward(self, input_states, headPositions=None):
        output = self.classifier(input_states)
        output = self.nonlinearity(output) if self.nonlinearity else output
        return output








