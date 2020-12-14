
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------------------------------#
# ROUTE

class Route(nn.Module):
    def __init__(self, D_h=None, D_in=10, D_out=10):
        super(Route, self).__init__()
        if D_h is None: D_h = D_in
        self._x_lin = nn.Linear(D_in, D_out, bias=False)
        self._h_dot = nn.Linear(D_h,  1,     bias=False)
        self._x_dot = nn.Linear(D_in, 1,     bias=False)
        self._g = 0

    def forward(self, h, x):
        g = self._x_dot(x)
        if h is not None: g = self._h_dot(h) + g
        g = torch.sigmoid(g)
        self._g = g.mean().item()
        y = g * self._x_lin(x)
        return y

    def candidness(self):
        return self._g


# ----------------------------------------------------------------------------------------------------------------------#
# SOURCE

class Source(nn.Module):
    def __init__(self, D_in, D_out):
        super(Source, self).__init__()
        self._x_lin = nn.Linear(D_in, D_out, bias=False)

    def forward(self, x):
        #if 2 in [p._version for p in self._x_lin.parameters()]:
        #    print('Version change in "Source":', [p._version for p in self._x_lin.parameters()])
        return self._x_lin(x)

    def candidness(self): return -1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING :
