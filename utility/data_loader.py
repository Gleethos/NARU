
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


def load_jokes():
    df = pd.read_csv('../data/jokes/reddit.csv', index_col='ID', header=0)
    jokes = df['Joke'].to_list()
    jokes = [tokenize(j) for j in jokes]
    return jokes

