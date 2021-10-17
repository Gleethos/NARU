import os.path

from lib.embedding import Encoder
import torch
from lib.model.ffnaru import Network
from lib.data_loader import load_jokes
from lib.trainer import exec_trial_with_autograd
from lib.model.persist import save_params
from lib.model.comps.classes import CONTEXT

# ---------------------------------------------------------------------

from scipy.spatial import KDTree


class TestEncoder:

    def __init__(self):
        self.word_to_vec = {
            'a': torch.tensor([[1, 1]], dtype=torch.float32),
            'c': torch.tensor([[-1, -1]], dtype=torch.float32),
            'g': torch.tensor([[-1, 1]], dtype=torch.float32),
            't': torch.tensor([[1, -1]], dtype=torch.float32),
        }
        self.vec_i_to_word = dict()
        vectors, i = [], 0
        for word, vector in self.word_to_vec.items():
            vectors.append(vector.view(-1).tolist())
            self.vec_i_to_word[i] = word
            i = i + 1
        self.tree = KDTree(vectors)

    def sequence_words_in(self, seq):
        return [self.word_to_vec[w] for w in seq]

    def sequence_vecs_in(self, seq):
        result = []
        for vec in seq:
            d, i = self.tree.query(vec.view(-1).tolist())
            result.append(self.vec_i_to_word[i])
        return result


def test_with_autograd_on_dummy_data():
    CONTEXT.BPTT_limit = 10  # 10
    model = Network(  # feed-forward-NARU
        depth=4,
        max_height=3,
        max_dim=4,
        max_cone=3,
        D_in=2,
        D_out=2,
        with_bias=False
    )
    for W in model.get_params(): W.requires_grad = True
    data = [
        't c a g c a g'.split(),
        'a g c g a t c'.split(),
        'c a c t a c a'.split(),
        'g t c a g c t'.split()
    ]
    optimizer = torch.optim.Adam(model.get_params(), lr=0.03)
    encoder = TestEncoder()
    for i in range(1):
        choice_matrices = exec_trial_with_autograd(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=data[:],
            test_data=data[:],
            epochs=30,
            make_plots=False,
            print_epochs=False
        )
        print(choice_matrices)

    # a : [-4,  2,  3]
    # b : [-1, -6,  4]
    # c : [ 2,  6, -4]
    pred = model.pred(encoder.sequence_words_in(data[0]))
    assert [x.tolist() for x in pred] == [[[-0.05607888102531433, -0.027415240183472633]],
                                          [[-0.08839356899261475, 0.05141408368945122]],
                                          [[-0.2587147057056427, -0.07983965426683426]],
                                          [[-0.25858554244041443, -0.08517489582300186]],
                                          [[-0.06308478862047195, 0.06079373136162758]],
                                          [[-0.22751617431640625, -0.05428335443139076]]]
    # print('\n'.join([str(p.tolist()) for p in pred]))
    # print('\n')
    pred = model.pred(encoder.sequence_words_in(data[1]))
    # print('\n'.join([str(p.tolist()) for p in pred]))

    assert [x.tolist() for x in pred] == [[[-0.057036757469177246, -0.02962394617497921]],
                                          [[-0.0841657742857933, 0.030372582376003265]],
                                          [[-0.08507544547319412, 0.028935125097632408]],
                                          [[-0.08338922262191772, 0.029478898271918297]],
                                          [[-0.25068309903144836, -0.09809699654579163]],
                                          [[-0.3227868378162384, -0.13274739682674408]]]
    print('FFNN-NARU UNIT TEST DONE!')


test_with_autograd_on_dummy_data()
