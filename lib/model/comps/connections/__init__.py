
import torch
from lib.model.comps.fun import sig, activation, gaus
from lib.model.comps import fun
from lib.model.comps import Moment


class AbstractRoute:
    def reset(self): pass

# -------------------------------------------------------------------------------------------------------------------- #
# ROUTE

class Route(AbstractRoute):

    def __init__(self, D_in=10, D_out=10):
        D_h = D_out
        self.Wr = torch.randn(D_in, D_out)
        self.Wr.grad = torch.zeros(D_in, D_out)

        self.Wgh = torch.randn(D_h, 1)
        self.Wgh.grad = torch.zeros(D_h, 1)

        self.Wgr = torch.randn(D_in, 1)
        self.Wgr.grad = torch.zeros(D_in, 1)

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
        rec[self] = m
        g = ( r @ self.Wgr + h @ self.Wgh ) / 2
        m.pd_g = sig(g, derive=True) / 2 # inner times outer derivative
        g = sig(g)
        if not 0 <= g <= 1: print('Illegal gate:', g)
        assert 0 <= g <= 1  # Sigmoid should be within bounds!
        m.g = g.mean().item()  # g is used for routing! Largest gate wins!
        m.z = r @ self.Wr # z is saved for back-prop.
        return g * m.z # Returning vector "c", the gated connection vector!


# -------------------------------------------------------------------------------------------------------------------- #
# FAT ROUTE

class FatRoute(AbstractRoute):

    def __init__(self, D_in=10, D_out=10):
        D_h = D_out
        D_rh = D_in + D_h
        self.Wrh0      = torch.randn(D_rh, D_rh, requires_grad=True)
        self.Wrh0.grad = torch.zeros(D_rh, D_rh)
        self.Wrh       = torch.randn(D_rh, D_out, requires_grad=True)
        self.Wrh.grad  = torch.zeros(D_rh, D_out)

        self.Wgrh0      = torch.randn(D_rh, D_rh, requires_grad=True)
        self.Wgrh0.grad = torch.zeros(D_rh, D_rh)
        self.Wgrh       = torch.randn(D_rh, D_out, requires_grad=True)
        self.Wgrh.grad  = torch.zeros(D_rh, D_out)

    def get_params(self): return [ self.Wrh0, self.Wrh, self.Wgrh0, self.Wgrh ]

    def set_params(self, params: list):
        grad1, grad2, grad3, grad4 = self.Wrh0.grad, self.Wrh.grad, self.Wgrh0.grad, self.Wgrh.grad
        self.Wrh0  *= 0
        self.Wrh   *= 0
        self.Wgrh0 *= 0
        self.Wgrh  *= 0
        self.Wrh0  += params.pop(0)
        self.Wrh   += params.pop(0)
        self.Wgrh0 += params.pop(0)
        self.Wgrh  += params.pop(0)
        self.Wrh0.grad  = grad1
        self.Wrh.grad   = grad2
        self.Wgrh0.grad = grad3
        self.Wgrh.grad  = grad4

    def forward(self, h: torch.Tensor, r: torch.Tensor, rec: dict):
        m = Moment()
        rec[self] = m
        rh0 = torch.cat((r, h), dim=1)
        # Note: Now we need a capped activation function so that gradients don't explode
        # Also: sigmoid converges better than tanh.
        rh = fun.sig(rh0 @ self.Wrh0) @ self.Wrh
        # Note: GaSU us better than GaTU here!
        rhg = fun.gaus( fun.gatu(rh0.detach() @ self.Wgrh0) @ self.Wgrh )
        m.g = rhg.mean().item() # g is used for routing! Largest gate wins!
        m.z = rhg * rh # z is saved for back-prop.
        return m.z # Returning vector "c", the gated connection vector!


# -------------------------------------------------------------------------------------------------------------------- #
# FAT LSTM ROUTE

class FatLSTMRoute(AbstractRoute):

    def __init__(self, D_in=10, D_out=10):
        D_h = D_out
        D_rh = D_in + D_h
        self.Wrh0      = torch.randn(D_rh, D_out, requires_grad=True)
        self.Wrh0.grad = torch.zeros(D_rh, D_out)

        self.Wgrh0      = torch.randn(D_rh, D_rh, requires_grad=True)
        self.Wgrh0.grad = torch.zeros(D_rh, D_rh)
        self.Wgrh       = torch.randn(D_rh, D_out, requires_grad=True)
        self.Wgrh.grad  = torch.zeros(D_rh, D_out)
        self.lstm = LSTMCell(input_size=D_out, hidden_size=D_out)

    def reset(self):
        self.lstm.reset()

    def get_params(self): return [ self.Wrh0, self.Wgrh0, self.Wgrh ]

    def set_params(self, params: list):
        grad1, grad3, grad4 = self.Wrh0.grad, self.Wgrh0.grad, self.Wgrh.grad
        self.Wrh0  *= 0
        self.Wgrh0 *= 0
        self.Wgrh  *= 0
        self.Wrh0  += params.pop(0)
        self.Wgrh0 += params.pop(0)
        self.Wgrh  += params.pop(0)
        self.Wrh0.grad  = grad1
        self.Wgrh0.grad = grad3
        self.Wgrh.grad  = grad4

    def forward(self, h: torch.Tensor, r: torch.Tensor, rec: dict):
        m = Moment()
        rec[self] = m
        rh0 = torch.cat((r, h), dim=1)
        # Note: Now we need a capped activation function so that gradients don't explode
        # Also: sigmoid converges better than tanh.
        #rh = fun.sig(rh0 @ self.Wrh0) @ self.Wrh
        rh = self.lstm.call(fun.sig(rh0 @ self.Wrh0))
        # Note: GaSU us better than GaTU here!
        rhg = fun.gaus( fun.gatu(rh0.detach() @ self.Wgrh0) @ self.Wgrh )
        m.g = rhg.mean().item() # g is used for routing! Largest gate wins!
        m.z: torch.Tensor = rhg * rh # z is saved for back-prop.
        return m.z# Returning vector "c", the gated connection vector!


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = torch.randn(4 * hidden_size, input_size, requires_grad=True)
        self.weight_hh = torch.randn(4 * hidden_size, hidden_size, requires_grad=True)
        self.bias_ih = torch.randn(4 * hidden_size, requires_grad=True)
        self.bias_hh = torch.randn(4 * hidden_size, requires_grad=True)
        self.h = None
        self.c = None

    def call(self, input):
        if self.h is None:
            self.h = input
            self.c = input

        self.h, self.c = self.forward(input, (self.h, self.c))
        return self.h

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

    def reset(self):
        self.h = None
        self.c = None


# -------------------------------------------------------------------------------------------------------------------- #
# DEEP ROUTE

class DeepRoute(AbstractRoute):

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

    def get_params(self):
        return [
            self.Wr0, self.Wr, self.Wgr0, self.Wgr, self.Wgh0, self.Wgh
        ]

    def set_params(self, params: list):
        grad1, grad2, grad3, grad4, grad5, grad6 = self.Wr0.grad, self.Wr.grad, self.Wgr0.grad, self.Wgr.grad, self.Wgh0.grad, self.Wgh.grad
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
        r0, h0 = activation(r.detach() @ self.Wgr0), activation(h.detach() @ self.Wgh0)
        g = ( r0 @ self.Wgr + h0 @ self.Wgh ) / 2
        m.pd_g = sig(g, derive=True) / 2 # inner times outer derivative
        g = sig(g+2)
        if not 0 <= g <= 1: print('Illegal gate:', g)
        assert 0 <= g <= 1  # Sigmoid should be within bounds!
        m.g = g.mean().item() # g is used for routing! Largest gate wins!
        m.z = (activation(r @ self.Wr0)) @ self.Wr # z is saved for back-prop.
        rec[self] = m
        return g * m.z # Returning vector "c", the gated connection vector!



# -------------------------------------------------------------------------------------------------------------------- #
# DEEP SMART ROUTE


class DeepSmartRoute(AbstractRoute):

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
        grad1, grad2, grad3, grad4, grad5, grad6 = self.Wr0.grad, self.Wr.grad, self.Wgr0.grad, self.Wgr.grad, self.Wgh0.grad, self.Wgh.grad
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
        r0, h0 = activation(r.detach() @ self.Wgr0), activation(h.detach() @ self.Wgh0) # sig is good here!
        g = ( r0 @ self.Wgr + h0 @ self.Wgh ) / 2
        m.pd_g = sig(g, derive=True) / 2 # inner times outer derivative
        g = gaus(g, mean=0, std=1)
        if not 0 <= g <= 1: print('Illegal gate:', g)
        assert 0 <= g <= 1  # Sigmoid should be within bounds!
        m.g = g.mean().item()  # g is used for routing! Largest gate wins!
        #print(m.g)
        #m.z = r @ self.Wr # z is saved for back-prop.
        #m.z = torch.softmax(r @ self.Wr0, dim=1, dtype=torch.float32) @ self.Wr # z is saved for back-prop.
        m.z = (sig(r @ self.Wr0)) @ self.Wr # z is saved for back-prop.
        rec[self] = m
        return m.z*g # Returning vector "c", the gated connection vector!


# -------------------------------------------------------------------------------------------------------------------- #
# BIASED DEEP SMART ROUTE


class BiasedDeepSmartRoute(AbstractRoute):

    def __init__(self, D_in=10, D_out=10):
        D_h = D_out

        self.Wr0 = torch.randn(D_in, D_in)
        self.Wr0.grad = torch.zeros(D_in, D_in)
        self.br = torch.randn(1, D_in)
        self.br.grad = torch.zeros(1, D_in)
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
            self.Wr0, self.br, self.Wr, self.Wgr0, self.Wgr, self.Wgh0, self.Wgh
        ]

    def set_params(self, params: list):
        grad1, grad2, grad3, grad4, grad5, grad6, grad7 = self.Wr0.grad, self.br, self.Wr.grad, self.Wgr0.grad, self.Wgr.grad, self.Wgh0.grad, self.Wgh.grad
        self.Wr0  *= 0
        self.br   *= 0
        self.Wr   *= 0
        self.Wgr0 *= 0
        self.Wgr  *= 0
        self.Wgh0 *= 0
        self.Wgh  *= 0
        self.Wr0  += params.pop(0)
        self.br  += params.pop(0)
        self.Wr   += params.pop(0)
        self.Wgr0 += params.pop(0)
        self.Wgr  += params.pop(0)
        self.Wgh0 += params.pop(0)
        self.Wgh  += params.pop(0)
        self.Wr0.grad  = grad1
        self.br.grad   = grad2
        self.Wr.grad   = grad3
        self.Wgr0.grad = grad4
        self.Wgr.grad  = grad5
        self.Wgh0.grad = grad6
        self.Wgh.grad  = grad7

    def forward(self, h: torch.Tensor, r: torch.Tensor, rec: dict):
        m = Moment()
        r0, h0 = activation(r.detach() @ self.Wgr0), activation(h.detach() @ self.Wgh0) # sig is good here!
        g = ( r0 @ self.Wgr + h0 @ self.Wgh ) / 2
        m.pd_g = sig(g, derive=True) / 2 # inner times outer derivative
        g = gaus(g, mean=0, std=1)
        if not 0 <= g <= 1: print('Illegal gate:', g)
        assert 0 <= g <= 1  # Sigmoid should be within bounds!
        m.g = g.mean().item()  # g is used for routing! Largest gate wins!
        #print(m.g)
        #m.z = r @ self.Wr # z is saved for back-prop.
        #m.z = torch.softmax(r @ self.Wr0, dim=1, dtype=torch.float32) @ self.Wr # z is saved for back-prop.
        m.z = (sig((r @ self.Wr0)+self.br)) @ self.Wr # z is saved for back-prop.
        rec[self] = m
        return m.z*g # Returning vector "c", the gated connection vector!


