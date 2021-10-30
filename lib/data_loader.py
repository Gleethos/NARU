
# Data src:
# https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm

import pandas as pd
from spacy.lang.en import English

parser = English()
print(parser)


def tokenize(text):
    tokens = parser(text)
    tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
    return tokens


def load_jokes(prefix=''):
    df = pd.read_csv(prefix+'data/jokes/reddit.csv', index_col='ID', header=0)
    jokes = df['Joke'].to_list()
    jokes = [tokenize(j) for j in jokes]
    return jokes

def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return list_to_split[:middle], list_to_split[middle:]

