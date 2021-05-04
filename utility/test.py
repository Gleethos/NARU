
from utility.embedding import Encoder
import torch
from utility.ffnaru import Network
from utility.data_loader import load_jokes
from utility.trainer import exec_trial_with_autograd
from utility.persistence import save_params
from utility.classes import CONTEXT, Route
import time

# ---------------------------------------------------------------------


def test_with_autograd_on_jokes():# Uses PyTorchs auto-grad:
    CONTEXT.BPTT_limit = 10 #10

    model = Network( # feed-forward-NARU
        depth=7,
        max_height=18,
        max_dim=128,
        max_cone=6,
        D_in=50,
        D_out=50,
        with_bias=False
    )
    for W in model.get_params(): W.requires_grad = True

    jokes = load_jokes() # Total: 1592
    optimizer = torch.optim.Adam(model.get_params(), lr=3e-4)
    encoder = Encoder()

    print(model.str())
    print()
    print(jokes[:10])
    print()
    print(jokes[10:15])

    #save_params( [torch.range(0, 10, 3)], 'models/hey/' )
    #print(load_params('models/hey/'))
    #save_params( [torch.range(0, 10, 3)*17-10], 'models/hey/' )
    #print(load_params('models/hey/'))

    #model.set_params(load_params('models/test_model/'))

    for i in range(1):
        target_folder = 'models/test_model/directed-NARU-net_' + time.strftime("%Y%m%d-%H%M%S") + '/'
        choice_matrices = exec_trial_with_autograd(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=jokes[100:],
            test_data=jokes[0:100],
            epochs=200,
            batch_size=10,
            path=target_folder
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
            'a':torch.tensor([[-4.0, 2.0, 3.0]], dtype=torch.float32),
            'b': torch.tensor([[-1.0, -6.0, 4.0]], dtype=torch.float32),
            'c': torch.tensor([[2.0, 6.0, -4.0]], dtype=torch.float32),
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
        D_in=3,
        D_out=3,
        with_bias=False
    )
    for W in model.get_params(): W.requires_grad = True
    data = ['c c a b c a'.split(), 'a b c b a c'.split(), 'c a c c a b'.split(), 'b a c a b c'.split()]
    optimizer = torch.optim.Adam(model.get_params(), lr=0.1)
    encoder = TestEncoder()
    for i in range(1):
        choice_matrices = exec_trial_with_autograd(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=data[:4],
            test_data=data[:4],
            epochs=35
        )
        print(choice_matrices)

    # a : [-4,  2,  3]
    # b : [-1, -6,  4]
    # c : [ 2,  6, -4]
    pred = model.pred(encoder.sequence_words_in('a b c b a c'.split()))
    print('\n'.join([str(p.tolist()) for p in pred]))
    print('\n')
    pred = model.pred(encoder.sequence_words_in('c c a b c a'.split()))
    print('\n'.join([str(p.tolist()) for p in pred]))
    print('FFNN-NARU TEST DONE!')

    test_sentence = encoder.sequence_words_in('a b c b a c'.split())
    preds = model.pred(test_sentence)
    print(' '.join(encoder.sequence_vecs_in(preds)))
    test_sentence = encoder.sequence_words_in('c c a b c a'.split())
    preds = model.pred(test_sentence)
    print(' '.join(encoder.sequence_vecs_in(preds)))



#test_with_autograd_on_jokes()
test_with_autograd_on_dummy_data()
