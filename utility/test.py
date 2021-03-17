
from utility.embedding import Encoder
import torch
from utility.ffnaru import Network
from utility.data_loader import load_jokes
from utility.trainer import exec_trial_with_autograd
from utility.persistence import save_params, load_params
from utility.classes import CONTEXT, Route


# ---------------------------------------------------------------------

# Uses PyTorchs auto-grad:
def test_with_autograd_on_jokes():
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
    for W in model.get_params(): W.requires_grad = True

    jokes = load_jokes()
    optimizer = torch.optim.Adam(model.get_params(), lr=0.01)
    encoder = Encoder()

    #save_params( [torch.range(0, 10, 3)], 'models/hey/' )
    #print(load_params('models/hey/'))
    #save_params( [torch.range(0, 10, 3)*17-10], 'models/hey/' )
    #print(load_params('models/hey/'))

    #model.set_params(load_params('models/test_model/'))

    for i in range(1):
        choice_matrices = exec_trial_with_autograd(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=jokes[:10],
            test_data=jokes[10:15],
            epochs=200
        )
        print(choice_matrices)
        # SAVING PARAMETERS:
        #target_folder = 'models/test_model/' # 'models/feed-forward-NARU_'+time.strftime("%Y%m%d-%H%M%S")+'/'
        #save_params( model.get_params(), target_folder )


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



def test_custom_backprop_on_dummy_data():
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
    data = ['c a b c c'.split(), 'b c b a c'.split()]
    optimizer = torch.optim.Adam(model.get_params(), lr=0.1)
    encoder = TestEncoder()
    for i in range(1):
        choice_matrices = exec_trial_with_autograd(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=data[:1],
            test_data=data[:1],
            epochs=500,
            use_custom_backprop=True
        )
        print(choice_matrices)
    print('FFNN-NARU TEST DONE!')



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
    data = ['c a b c c'.split(), 'b c b a c'.split()]
    optimizer = torch.optim.Adam(model.get_params(), lr=0.1)
    encoder = TestEncoder()
    for i in range(1):
        choice_matrices = exec_trial_with_autograd(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            training_data=data[:2],
            test_data=data[:2],
            epochs=500
        )
        print(choice_matrices)
    pred = model.pred(encoder.sequence_words_in('b c b a c'.split()))
    print('\n'.join([str(p.tolist()) for p in pred]))
    print('FFNN-NARU TEST DONE!')



def test_custom_mse():
    test_c = torch.tensor([[0.2006, -0.0495]])
    test_c.requires_grad = True
    loss = torch.nn.MSELoss()
    l = loss(input=test_c, target=torch.tensor([[-1.0, 1.0]]))
    # mse = ((c-t)^2)/2
    l.backward()
    assert str(test_c.grad) == str(torch.tensor([[0.2006, -0.0495]]) - torch.tensor([[-1.0, 1.0]]))


def test_custom_backprop_for_Route():
    test_custom_mse()
    torch.manual_seed(66642999)

    route = Route(D_in=3, D_out=2)
    r, h = torch.ones(1, 3), torch.ones(1, 2)

    rec = dict()

    for W in route.get_params(): W.requires_grad = True
    h.requires_grad = True
    r.requires_grad = True

    #c.requires_grad = True
    #assert str(c) == 'tensor([[ 0.2006, -0.0495]], grad_fn=<CloneBackward>)'

    c = route.forward(h=h, r=r, rec=rec)
    loss = torch.nn.MSELoss()
    l = loss(input=c, target=torch.tensor([[-1.0, 1.0]]))
    l.backward()

    print(str(c-torch.tensor([[-1.0, 1.0]])))
    print(str(h.grad.tolist()  ))
    print(str(r.grad.tolist()  ))
    print(str(route.Wgh.grad.tolist() ))
    print(str(route.Wgr.grad.tolist() ))
    print(str(route.Wr.grad .tolist() ))
    assert str(h.grad.tolist())            == '[[0.06240329518914223, -0.11417414247989655]]'
    assert str(r.grad.tolist())            == '[[-0.5525853633880615, 0.9538493156433105, -0.10835178941488266]]'
    assert str(route.Wgh.grad.tolist()) == "[[0.08430635929107666], [0.08430635929107666]]"
    assert str(route.Wgr.grad.tolist()) == "[[0.08430635929107666], [0.08430635929107666], [0.08430635929107666]]"
    assert str(route.Wr.grad .tolist()) == "[[0.5091030597686768, -0.44506365060806274], [0.5091030597686768, -0.44506365060806274], [0.5091030597686768, -0.44506365060806274]]"

    for W in route.get_params(): W.grad = W.grad * 0

    g_h, g_r = route.backward(
        e_c=(c - torch.tensor([[-1.0, 1.0]])),
        rec=rec,
        h=torch.tensor([[2.0, -3.0]], dtype=torch.float32),
        r=torch.tensor([[-1.0, 4.0, 2.0]], dtype=torch.float32)
    )
    assert str(g_h.tolist())            == '[[0.06240329518914223, -0.11417414247989655]]'
    assert str(g_r.tolist())            == '[[-0.5525853633880615, 0.9538493156433105, -0.10835178941488266]]'
    assert str(route.Wgh.grad.tolist()) == "[[0.08430635929107666], [0.08430635929107666]]"
    assert str(route.Wgr.grad.tolist()) == "[[0.08430635929107666], [0.08430635929107666], [0.08430635929107666]]"
    assert str(route.Wr.grad .tolist()) == "[[0.5091030597686768, -0.44506365060806274], [0.5091030597686768, -0.44506365060806274], [0.5091030597686768, -0.44506365060806274]]"

#test_with_autograd_on_jokes()
#test_custom_backprop_for_Route()
#test_custom_backprop_on_dummy_data()
test_with_autograd_on_dummy_data()
