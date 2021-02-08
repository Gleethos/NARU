
# Data src:
# https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm

import pandas as pd


def load_jokes():
    df = pd.read_csv('data/jokes/reddit.csv', index_col='ID', header=0)
    return df['Joke'].to_list()

