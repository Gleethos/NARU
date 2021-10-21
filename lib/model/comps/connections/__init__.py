
import torch
from lib.model.comps.fun import sig, activation
from lib.model.comps import CONTEXT
from lib.model.comps import Moment


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

    def set_params(self, params: list):
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

#---


# -------------------------------------------------------------------------------------------------------------------- #
# ROUTE

class DeepRoute:

    def __init__(self, D_in=10, D_out=10):
        D_h = D_out

        self.Wr0 = torch.randn(D_in, D_in)
        self.Wr0.grad = torch.zeros(D_in, D_in)
        self.Wr = torch.randn(D_in, D_out)
        self.Wr.grad = torch.zeros(D_in, D_out)

        self.Wgh0 = torch.randn(D_h, D_h)
        self.Wgh0.grad = torch.zeros(D_h, D_h)
        self.Wgh = torch.randn(D_h, 1)
        self.Wgh.grad = torch.zeros(D_h, 1)

        self.Wgr0 = torch.randn(D_in, D_in)
        self.Wgr0.grad = torch.zeros(D_in, D_in)
        self.Wgr = torch.randn(D_in, 1)
        self.Wgr.grad = torch.zeros(D_in, 1)

        self.pd_g: torch.Tensor = None  # the partial derivative of g

    def get_params(self):
        return [
            self.Wr0, self.Wr, self.Wgr0, self.Wgr, self.Wgh0, self.Wgh
        ]

    def set_params(self, params: list):
        grad1, grad2, grad3, grad4, grad5, grad6 = self.Wr0, self.Wr.grad, self.Wgr0, self.Wgr.grad, self.Wgh0, self.Wgh.grad
        self.Wr0  *= 0
        self.Wr   *= 0
        self.Wgr0 *= 0
        self.Wgr  *= 0
        self.Wgh0 *= 0
        self.Wgh  *= 0
        self.Wr0  += params.pop(0)
        self.Wr   += params.pop(0)
        self.Wgr0 += params.pop(0)
        self.Wgr  += params.pop(0)
        self.Wgh0 += params.pop(0)
        self.Wgh  += params.pop(0)
        self.Wr0.grad  = grad1
        self.Wr.grad   = grad2
        self.Wgr0.grad = grad3
        self.Wgr.grad  = grad4
        self.Wgh0.grad = grad5
        self.Wgh.grad  = grad6

    def forward(self, h: torch.Tensor, r: torch.Tensor, rec: dict):
        m = Moment()
        r0, h0 = activation(r @ self.Wgr0), activation(h @ self.Wgh0)
        g = ( r0 @ self.Wgr + h0 @ self.Wgh ) / 2
        m.pd_g = sig(g, derive=True) / 2 # inner times outer derivative
        g = sig(g)
        if not 0 <= g <= 1: print('Illegal gate:', g)
        assert 0 <= g <= 1  # Sigmoid should be within bounds!
        m.g = g.mean().item()  # g is used for routing! Largest gate wins!
        m.z = activation(r @ self.Wr0) @ self.Wr # z is saved for back-prop.
        rec[self] = m
        return g * m.z # Returning vector "c", the gated connection vector!


