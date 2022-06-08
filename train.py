from pyhocon import ConfigFactory
import argparse
from taskManager import TaskManager

import numpy as np
import pandas as pd

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help='Filename of the model.', nargs='+')
    parser.add_argument('--train', action='store_true', help='Set the code to training purpose.')
    parser.add_argument('--config', type=str, help='Filename of the configuration.')
    args = parser.parse_args()

    if args.train:
        config = ConfigFactory.parse_file(args.config)
        taskManager = TaskManager(config, 1234)

        train_df = read_sents(taskManager[0].trainSentences)

        print (train_df)
