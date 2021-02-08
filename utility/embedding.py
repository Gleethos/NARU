
# This approach is based on the following:
# https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
# GloVe:
# https://nlp.stanford.edu/projects/glove/

import numpy as np
import pickle
import bcolz as bcolz
from collections import defaultdict
import hashlib

glove_path = 'embedding_data'


def convert_txt_to_pkl_files():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

    with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))


class keydefaultdict(defaultdict):
    def __missing__(self, seed):
        np.random.seed(int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16) % (10 ** 8))
        return np.random.randn(50)


def load_word_vec_dict():
    # Using those objects we can now create a dictionary that given a word returns its vector.
    vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    glove = keydefaultdict(None, glove)
    return glove
