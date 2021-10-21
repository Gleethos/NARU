
import torch

from lib.model.comps import CONTEXT
from lib.model.comps.connections import Route

CONTEXT.routeClass = Route

from lib.embedding import Encoder
from lib.model.ffnaru import Network
from lib.data_loader import load_jokes
from lib.trainer import exec_trial_with_autograd
from lib.model.persist import save_params

print('Route loaded! Unit-Testing now...')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING:

def test_route():
    torch.manual_seed(66642999)
    CONTEXT.recorders = []

    route = Route(D_in=3, D_out=2)

    assert len(CONTEXT.recorders) == 0
    #assert CONTEXT.recorders[0] == route

    r, h = torch.ones(1, 3), torch.ones(1, 2)

    rec = dict()

    c = route.forward(h, r, rec)

    assert str(c) == 'tensor([[ 0.2006, -0.0495]])'

    del route, r, h#, c, g_r, g_h
    CONTEXT.recorders = []
    print('Route Unit-Testing successful!')


test_route()
#---


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
    torch.manual_seed(66642999)
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
        #print(choice_matrices)

    # a : [-4,  2,  3]
    # b : [-1, -6,  4]
    # c : [ 2,  6, -4]
    pred = model.pred(encoder.sequence_words_in(data[0]))

    assert [x.tolist() for x in pred] == [[[-0.055136412382125854, -0.03600181266665459]], [[-0.05615430697798729, -0.03751407936215401]], [[-0.398160457611084, 0.13682778179645538]], [[-0.5601562857627869, 0.016082024201750755]], [[0.045633673667907715, -0.11803268641233444]], [[0.16237783432006836, -0.02257206104695797]]]
    # print('\n'.join([str(p.tolist()) for p in pred]))
    # print('\n')
    pred = model.pred(encoder.sequence_words_in(data[1]))
    # print('\n'.join([str(p.tolist()) for p in pred]))
    assert [x.tolist() for x in pred] == [[[-0.5060637593269348, -0.40269386768341064]], [[-0.682539701461792, -0.5389066934585571]], [[-0.7031692862510681, -0.5533317923545837]], [[-0.030585220083594322, 0.07828956097364426]], [[-0.6425731182098389, -0.48571181297302246]], [[-0.8725153803825378, -0.7259251475334167]]]
    print('FFNN-NARU UNIT TEST DONE!')

def test_with_autograd_on_dummy_data_2():
    torch.manual_seed(66642999)
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
            epochs=5,
            make_plots=False
        )
        print(choice_matrices)

    # a : [-4,  2,  3]
    # b : [-1, -6,  4]
    # c : [ 2,  6, -4]
    pred = model.pred(encoder.sequence_words_in(data[0]))
    print('\n'.join([str(p.tolist()) for p in pred]))
    print('\n')
    pred = model.pred(encoder.sequence_words_in(data[1]))
    print('\n'.join([str(p.tolist()) for p in pred]))
    print('FFNN-NARU TEST DONE!')
    print([x.tolist() for x in pred])
    assert [x.tolist() for x in pred] == [[[-0.010608508251607418, -0.00998143944889307]],
                                            [[-0.008684919215738773, -0.010771256871521473]],
                                            [[-0.011129743419587612, -0.009816867299377918]],
                                            [[-0.007469087839126587, 0.0017242614412680268]],
                                            [[-0.01924082823097706, -0.009243465960025787]],
                                            [[-0.0158716831356287, -0.009900451637804508]]]

    for s in data:
        test_sentence = encoder.sequence_words_in(s)
        preds = model.pred(test_sentence)
        print(' '.join(s),':',' '.join(encoder.sequence_vecs_in(preds)))

test_with_autograd_on_dummy_data()
test_with_autograd_on_dummy_data_2()
print('\nNaru main package successfully loaded!')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')