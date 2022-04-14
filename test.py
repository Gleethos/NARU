
import torch

from lib.embedding import Encoder
from lib.model.ffnaru import Network
from lib.data_loader import load_jokes, list_splitter
from lib.trainer import exec_trial_with_autograd
from lib.model.persist import save_params
from lib.model import Settings
from lib.model.comps.connections import Route, DeepRoute, DeepSmartRoute, BiasedDeepSmartRoute
import time

# ---------------------------------------------------------------------


def test_with_autograd_on_jokes(path_prefix=''):# Uses PyTorchs auto-grad:
    torch.manual_seed(42)
    model = Network( # feed-forward-NARU
        depth=7,
        max_height=18,
        max_dim=128,
        max_cone=6,
        D_in=50,
        D_out=50,
        with_bias=False,
        settings=Settings(route=DeepRoute)
    )
    for W in model.get_params(): W.requires_grad = True

    jokes = load_jokes(prefix=path_prefix) # Total: 1592
    optimizer = torch.optim.Adam(model.get_params(), lr=3e-4)
    encoder = Encoder(path_prefix=path_prefix)

    training_data, test_data = list_splitter(jokes, 0.8)

    # Uncomment this if you want to merely play around:
    #training_data = jokes[:4]
    #test_data = jokes[4:8]

    #save_params( [torch.range(0, 10, 3)], 'models/hey/' )
    #print(load_params('models/hey/'))
    #save_params( [torch.range(0, 10, 3)*17-10], 'models/hey/' )
    #print(load_params('models/hey/'))

    #model.set_params(load_params('models/test_model/'))

    for i in range(1):
        target_folder = 'models/test_model_DR_B32/directed-NARU-net_' + time.strftime("%Y%m%d-%H%M%S") + '/'
        choice_matrices = exec_trial_with_autograd(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=training_data,
            test_data=test_data,
            epochs=500,
            batch_size=32,
            path=target_folder,
            do_ini_full_batch=False#
        )
        print('Latest choice matrices:', choice_matrices)
        # SAVING PARAMETERS:
        save_params( model, target_folder )


    print('Training procedure completed!')
    print('Testing now...')

    test_sentence = encoder.sequence_words_in('What did the bartender say to the jumper cables ?'.split())
    preds = model.pred(test_sentence)
    print('Model predicted:')
    print('"'+' '.join(encoder.sequence_vecs_in(preds))+'"')
    print('Should be:')
    print('"What did the bartender say to the jumper cables ?"')


from scipy.spatial import KDTree


class TestEncoder:

    def __init__(self):
        self.word_to_vec = {
            'a': torch.tensor([[ 1,  1]], dtype=torch.float32),
            'c': torch.tensor([[-1, -1]], dtype=torch.float32),
            'g': torch.tensor([[-1,  1]], dtype=torch.float32),
            't': torch.tensor([[ 1, -1]], dtype=torch.float32),
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
    model = Network(  # feed-forward-NARU
        depth=4,
        max_height=3,
        max_dim=4,
        max_cone=3,
        D_in=2,
        D_out=2,
        with_bias=False,
        settings=Settings(route=DeepRoute)
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
            epochs=600,
            make_plots=True
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

    for s in data:
        test_sentence = encoder.sequence_words_in(s)
        preds = model.pred(test_sentence)
        print(' '.join(s),':',' '.join(encoder.sequence_vecs_in(preds)))

test_with_autograd_on_jokes(path_prefix='../')
#test_with_autograd_on_dummy_data()

