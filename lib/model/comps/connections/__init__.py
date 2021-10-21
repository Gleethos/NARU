
import torch
from lib.model.comps.fun import sig
from lib.model.comps import CONTEXT
from lib.model.comps import Moment

torch.manual_seed(666)

# -------------------------------------------------------------------------------------------------------------------- #
# ROUTE

class Route:

    def __init__(self, D_in=10, D_out=10):
        D_h = D_out
        self.Wr = torch.randn(D_in, D_out)
        self.Wr.grad = torch.zeros(D_in, D_out)

        self.Wgh = torch.randn(D_h, 1)
        self.Wgh.grad = torch.zeros(D_h, 1)

        self.Wgr = torch.randn(D_in, 1)
        self.Wgr.grad = torch.zeros(D_in, 1)
        self.pd_g: torch.Tensor = None  # the partial derivative of g

    def get_params(self): return [self.Wr, self.Wgr, self.Wgh]

    def set_params(self, params):
        grad1, grad2, grad3 = self.Wr.grad, self.Wgr.grad, self.Wgh.grad
        self.Wr  *= 0
        self.Wgr *= 0
        self.Wgh *= 0
        self.Wr  += params.pop(0)
        self.Wgr += params.pop(0)
        self.Wgh += params.pop(0)
        self.Wr.grad = grad1
        self.Wgr.grad = grad2
        self.Wgh.grad = grad3

    def forward(self, h: torch.Tensor, r: torch.Tensor, rec: dict):
        m = Moment()
        g = ( r @ self.Wgr + h @ self.Wgh ) / 2
        m.pd_g = sig(g, derive=True) / 2 # inner times outer derivative
        g = sig(g)
        if not 0 <= g <= 1: print('Illegal gate:', g)
        assert 0 <= g <= 1  # Sigmoid should be within bounds!
        m.g = g.mean().item()  # g is used for routing! Largest gate wins!
        m.z = r @ self.Wr # z is saved for back-prop.
        rec[self] = m
        return g * m.z # Returning vector "c", the gated connection vector!



print('Route loaded! Unit-Testing now...')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING:

torch.manual_seed(66642999)

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

#---