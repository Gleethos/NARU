
import time
from utility.embedding import Encoder
import torch
from utility.ffnaru import Network
from utility.data_loader import load_jokes
from utility.trainer import exec_trial
from utility.persistence import save_params, load_params
from utility.classes import CONTEXT

# ---------------------------------------------------------------------
def test_1():

    CONTEXT.BPTT_limit = 10 #10

    model = Network( # feed-forward-NARU
        depth=5,
        max_height=10,
        max_dim=64,
        max_cone=5,
        D_in=50,
        D_out=50,
        with_bias=False
    )

    jokes = load_jokes()
    optimizer = torch.optim.Adam(model.get_params(), lr=0.0001)
    encoder = Encoder()

    #save_params( [torch.range(0, 10, 3)], 'models/hey/' )
    #print(load_params('models/hey/'))
    #save_params( [torch.range(0, 10, 3)*17-10], 'models/hey/' )
    #print(load_params('models/hey/'))

    #model.set_params(load_params('models/test_model/'))

    for i in range(1):
        choice_matrices = exec_trial(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=jokes[:1],
            test_data=jokes[10:15],
            epochs=20
        )
        print(choice_matrices)
        # SAVING PARAMETERS:
        target_folder = 'models/test_model/' # 'models/feed-forward-NARU_'+time.strftime("%Y%m%d-%H%M%S")+'/'
        save_params( model.get_params(), target_folder )


    print('FFNN-NARU TEST DONE!')

    test_sentence = encoder.sequence_words_in('What did the bartender say to the jumper cables ?'.split())
    preds = model.pred(test_sentence)
    print(' '.join(encoder.sequence_vecs_in(preds)))



class TestEncoder:

    def __init__(self):
        self.word_to_vec = {
            'a':torch.tensor([[-4.0, 2.0, 3.0]], dtype=torch.float32),
            'b': torch.tensor([[-1.0, -6.0, 4.0]], dtype=torch.float32),
            'c': torch.tensor([[2.0, 6.0, -4.0]], dtype=torch.float32),
        }

    def sequence_words_in(self, seq):
        return [self.word_to_vec[w] for w in seq]

    def sequence_vecs_in(self, seq):
        result = []
        return result



def test_2():

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
    print(model.str())

    data = ['a b c'.split(), 'c b a'.split()]
    optimizer = torch.optim.Adam(model.get_params(), lr=0.0001)
    encoder = TestEncoder()

    for i in range(1):
        choice_matrices = exec_trial(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=data[:1],
            test_data=data[:1],
            epochs=20
        )
        print(choice_matrices)

    print('FFNN-NARU TEST DONE!')

test_2()