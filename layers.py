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

    def forward(self, states, headPositions=None):

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








